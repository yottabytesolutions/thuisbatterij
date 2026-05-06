# Thuisbaterij-sim

Speelt de werkelijke load uit QuestDB plus PV af tegen historische EPEX
day-ahead en TenneT/ENTSO-E onbalansprijzen om batterij-, leveranciers- en
capaciteits-scenario's te benchmarken. Levert jaarkosten per scenario,
15-jarige mix-outlook en een capaciteits-gevoeligheidsanalyse met
vervangingen-aware TCO.

De brondata is verzameld met mijn meterlogger (zowel zon als zonnepanelen). 
Nog niet open source maar wel te gebruiken via
de [docker hub](https://hub.docker.com/repository/docker/yottabyte/meterlogger/general). 
Zelf draai ik het in k8s.

Dit is geschreven voor mijn eigen kosten/baten analyse. Dus vandaar questdb en enphase. Met een kleine aanpassing zou 
het met elke merk inverter met API ondersteuning moeten werken.

Omdat ik wat gaten had in mijn data (niet opgemerkte crashes) zit er ook een gap filler in welke op basis van werkelijk verbruik een fitting een realistische gap fill doet. Zie load_series in data.py.

## Setup

Build-systeem is `uv`. Installeer met `brew install uv` of
`curl -LsSf https://astral.sh/uv/install.sh | sh`.

```bash
cd sim
uv sync
cp config/user.example.toml config/user.toml # daarna invullen
```

`config/user.toml` bevat tarieven, contracten, batterij-defaults,
simulatieparameters, `questdb_url` en `entsoe_api_key`. Voor uitleg per veld:
zie `config/user.example.toml`.

### API keys

Voor volledige historische prijsdata zijn twee externe sleutels relevant:

- **ENTSO-E**: maak een account aan op het ENTSO-E Transparency Platform en
  vraag daar een API-token aan. Vul die in `config/user.toml` in als
  `entsoe_api_key`. Deze sleutel wordt gebruikt voor day-aheadprijzen en als
  primaire bron voor onbalansprijzen.
- **TenneT**: maak via het TenneT developer portal een API-key aan en zet die
  als omgevingsvariabele `TENNET_API_KEY`. Deze wordt alleen gebruikt als
  fallback voor onbalansprijzen wanneer ENTSO-E niet beschikbaar is.

Zonder sleutels kan de simulatie nog steeds draaien: day-ahead valt terug op de
publieke EnergyZero API en onbalansprijzen worden uiteindelijk synthetisch
gemaakt. De resultaten zijn dan minder geschikt voor nauwkeurige
onbalans-scenario's.

Prijsbron-fallback:

| Laag                         | Wanneer                | Bron                                   |
|------------------------------|------------------------|----------------------------------------|
| Day-ahead, primair           | `entsoe_api_key` gezet | ENTSO-E `query_day_ahead_prices('NL')` |
| Day-ahead, fallback          | geen sleutel           | EnergyZero publieke API                |
| Day-ahead, laatste redmiddel | beide niet             | gekalibreerd synthetisch               |
| Onbalans, primair            | `entsoe_api_key` gezet | ENTSO-E onbalans, mid van Long/Short   |
| Onbalans, fallback           | `TENNET_API_KEY` gezet | TenneT settlement-prices CSV           |
| Onbalans, laatste redmiddel  | geen                   | gesynthetiseerd uit day-ahead          |

## Draaien

```bash
uv run thuisbat-sim --start 2025-05-01 --end 2026-05-01

# capaciteits-sweep
uv run thuisbat-sim --start 2025-05-01 --end 2026-05-01 --sweep-capacity
```

Eerste run is traag (~2 min) door de QuestDB-pull en Numba-JIT-warmup.
Daarna 2-4 seconden door cache + JIT.

CLI-opties:

- `--start`, `--end`: venster (UTC). Default uit `config/user.toml`.
- `--capacity`: nominaal in kWh.
- `--max-power`: AC laad/ontlaadvermogen kW.
- `--config`: pad naar TOML-config (anders `config/user.toml`).
- `--sweep-capacity`: sweep over 5/8/10/12/15/20/25/30 kWh.
- `--monte-carlo N`: bootstrap N over alle gecachte historische jaren.
- `--workers`: aantal parallelle workers.

Output:

- `output/report.md`: jaarkosten per scenario en 15-jarige mix.
- `output/<scenario>.csv`: per-kwartier dispatch-trace.
- `output/sensitivity.md`: capaciteits-sweep.
- `output/monte_carlo.md`: Monte Carlo verdelingen.

## Scenario's

| Saldering-tijdperk                  | Post-saldering                          |
|-------------------------------------|-----------------------------------------|
| `baseline-fixed`                    | `baseline-fixed-postsaldering`          |
| `dynamic-no-battery`                | `dynamic-no-battery-postsaldering`      |
| `battery-pv-only`                   |                                         |
| `tibber-day-ahead`                  | `tibber-day-ahead-postsaldering`        |
| `tibber-lp-saldering` (LP-optimaal) | `tibber-lp-postsaldering` (LP-optimaal) |
| `frank-imbalance`                   | `frank-imbalance-postsaldering`         |
|                                     | `perfect-foresight-postsaldering`       |

## Ontwerp

- **Tijdsresolutie**: 15-min over het hele jaar. Day-ahead uurlijks,
  geforward-filled. ENTSO-E onbalans 15-min native; TenneT-CSV gereconstrueerd
  uit `Isp`.
- **Gap-fill**: schaal-vorm-model. Echte dagtotalen uit kWh-tellers, profiel
  per maand genormaliseerd op som 1, geschaald op het werkelijke dagtotaal.
  Ruwe QuestDB-data wordt nooit gewijzigd.
- **Batterij** (`battery.py`): SoC per kwartier, RT-rendement symmetrisch
  gesplitst (one-way = √RTE).
- **Dispatch** (`strategies.py`): per-kwartier-beslissing JIT-gecompileerd via
  Numba. Eerste call ~5 s warmup, daarna ~50× sneller dan pure Python.
- **Veroudering** (`aging.py`): LFP cycluslevensduur (6.000 cycli @ 80% DoD,
  0.5C, 25°C), DoD-exponent 1.7, C-rate-exponent 0.5, kalenderondergrens 14 jr.
- **Economie** (`economics.py`): commodity (uurlijks) versus vlakke kosten
  (energiebelasting, transport, BTW). Saldering tot 2027-01-01.
- **Onbalans-P&L**: symmetrisch. Slechte trades kosten ook geld.
- **Sweep**: `run_scenario(include_detail=False)` slaat de per-kwartier
  detail-DataFrame over.

## Cache

- **Nooit verwijderen** tenzij het schema wijzigt. ENTSO-E rate-limits maken
  opnieuw ophalen duur.
- Drie caches in `cache/`:
    - `load_<venster>.parquet`: 15-min verbruik/PV na gap-fill.
    - `<venster>_da.parquet` etc.: prijsfeeds.
    - `da_<COUNTRY>_<JAAR>.parquet`: meerjarige spread-analyse.
- Numba JIT-cache: `src/sim/__pycache__/_dispatch_loop_jit-*.nbi`.

## Verwachte layout QuestDB tabellen

### grid_meter

| Column                | Type      | Notes                 |
|-----------------------|-----------|-----------------------|
| `MeterMerkType`       | Symbol    |                       |
| `Serienummer`         | Symbol    |                       |
| `UsageCounter1`       | Double    | kWh off-peak          |
| `UsageCounter2`       | Double    | kWh peak              |
| `OutputCounter1`      | Double    | kWh returned off-peak |
| `OutputCounter2`      | Double    | kWh returned peak     |
| `VoltageP1/P2/P3`     | Double    | Volts                 |
| `TotalPowerUsage`     | Long      | W                     |
| `TotalPowerOutput`    | Long      | W                     |
| `BrownoutsP1/P2/P3`   | Long      |                       |
| `SpikesP1/P2/P3`      | Long      |                       |
| `CurrentP1/P2/P3`     | Long      | A                     |
| `PowerUsageP1/P2/P3`  | Long      | W                     |
| `PowerOutputP1/P2/P3` | Long      | W                     |
| `timestamp`           | Timestamp | From telegram.Time    |

### solar

| Column                | Type      | Notes                                      |
|-----------------------|-----------|--------------------------------------------|
| `EnvoySerialNumber`   | Symbol    |                                            |
| `ProductionWattHours` | Double    | Lifetime Wh                                |
| `ProductionWatt`      | Double    | Current W                                  |
| `ProductionVoltage`   | Long      | Contains PanelCount (naming is misleading) |
| `timestamp`           | Timestamp | From data.ReadingTime                      |
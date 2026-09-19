# Rebuilding the aligned records from the source archives

The raw model fields are redistributed by their providers under their own terms and are **not**
mirrored here. This file records the request parameters needed to rebuild
`data_processed/` from them. It is a specification, not a client: no retrieval code ships with this
repository, because none of it could be executed and verified in the environment the package was
assembled in, and an untested downloader in a reproduction package is worse than none.

Every parameter below is the one the reported analysis used. Where a value also appears in the
manuscript or supplement, the two agree.

---

## 1. Ground observations

    source        OpenAQ open-data archive, public Amazon S3 mirror (no API key required)
    pollutants    PM2.5, PM10
    resolution    hourly, as archived
    screening     sentinel (-999) and negative values removed
                  no upper concentration cap applied; the record's largest PM10 value
                  (2,978 ug/m3, E11 Road, 2023-09-15) is retained
    aggregation   averaged onto the three-hourly CAMS time grid
    network       19-station Abu Dhabi Emirate regulatory PM10 network, 2022-10-29 to 2023-12-31
                  (the benchmark arm adds five PM2.5 sites)

Station identifiers, coordinates, periods and matched sample counts are in Supplementary Table S1.

## 2. CAMS global reanalysis (EAC4) — the mechanistic field

    provider      Copernicus Atmosphere Data Store (ADS account required)
    dataset       CAMS global reanalysis (EAC4)
    variables     pm2p5   (surface PM2.5)
                  pm10    (surface PM10)
    level         surface
    grid          0.75 degrees
    time step     three-hourly
    units         retrieved in kg m-3; converted to ug m-3 by x 1e9
    matching      nearest grid cell to each station, by the rule
                    cell = (round(lat / 0.75), round(lon / 0.75))
                  Under this assignment the 19 stations occupy 9 cells; 15 of the 171 station
                  pairs share a cell, involving 16 of the 19 stations, at separations up to
                  approximately 46 km. `code/grid_offset.py` reproduces the assignment and its
                  sensitivity to the grid origin.

## 3. CAMS operational forecast — robustness comparison

    provider      Copernicus Atmosphere Data Store (ADS account required)
    dataset       CAMS global atmospheric composition forecasts
    variable      pm10, surface
    sampling      the same rolling origins as the main benchmark
    subset        matched late-2023 (September-December) dust period, 9,920 forecast points

Supplementary Table S14 and Figure S1 report this comparison.

## 4. MERRA-2 — independent-model cross-check

    provider      NASA GES DISC (Earthdata login required)
    collection    M2T1NXAER, version 5.12.4
    fields        hourly surface aerosol diagnostics
    matching      nearest grid cell, and nearest hour within a 90-minute window, to each
                  benchmark valid time
    coverage      9,872 of the 9,920 matched PM10 forecast points have a MERRA-2 value in window

Two reconstruction scenarios (PM2.5-mode and total-dust) are reported in Supplementary Table S19.

---

## What is already built

`data_processed/` holds the outputs of the steps above, so the audit and the simulation both run
without touching either archive:

    aligned_pm10_<station>.csv   19 PM10 series, three-hourly, observation and CAMS aligned
    aligned_<city>.csv           the five PM2.5 benchmark sites
    aligned_qc6/                 the same records under the flat-line rule K = 6 (reported)
    aligned_qc3/                 the same records under K = 3 (sensitivity)

`smoke_test.py` reloads these and re-checks the integrity invariants in under a minute; it prints
the matched sample count (20,818) that every empirical number in the manuscript rests on.

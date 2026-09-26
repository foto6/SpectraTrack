# Public pedestrian dataset import — MOT17 + CrowdHuman

A5 can reduce manual annotation by importing official public ground truth into the existing
`spectratrack.qa_benchmark` JSONL format.

This is **not** a second evaluator and public data does **not** replace the private CCTV holdout.

## Terms before download

### MOT17

MOTChallenge publishes its datasets under **CC BY-NC-SA 3.0**. Use is non-commercial,
attribution is required, and share-alike applies to adaptations.

Official page:

`https://motchallenge.net/data/MOT17/`

### CrowdHuman

CrowdHuman image data is limited to **non-commercial research/education** and the images
must **not** be redistributed.

Official page:

`https://www.crowdhuman.org/download.html`

The importer requires `--acknowledge-terms` so the generated import manifest records that
the user accepted the applicable dataset terms.

Dataset binaries remain under the gitignored local `benchmarks/vnext/qa/public/` directory.
Generated converted JSONL/import/freeze artifacts belong under the separate gitignored `benchmarks/vnext/qa/imports/` directory.

---

# 1. MOT17 train GT

## Download/layout

From `pc/benchmarks/vnext/qa/public/`:

```powershell
curl.exe -L https://motchallenge.net/data/MOT17.zip -o MOT17.zip
Expand-Archive .\MOT17.zip -DestinationPath .\MOT17
```

The effective `--dataset-root` must be the directory that directly contains `train/`.
Depending on archive extraction, that may be:

```text
benchmarks/vnext/qa/public/MOT17/
  train/
    MOT17-02-FRCNN/
      img1/
      gt/gt.txt
      seqinfo.ini
    MOT17-04-FRCNN/
    ...
```

If extraction creates one additional nested directory, point `--dataset-root` at that
nested directory instead.

The same seven physical train sequences are repeated for DPM/FRCNN/SDP detector-result
variants. A5 defaults to **FRCNN source directories** only so each physical frame sequence is
imported once.

## Import

From `pc/`:

```powershell
python -m spectratrack.vnext_qa import-mot17 `
  --dataset-root benchmarks/vnext/qa/public/MOT17 `
  --variant FRCNN `
  --output benchmarks/vnext/qa/imports/mot17-public.jsonl `
  --manifest benchmarks/vnext/qa/imports/mot17-public.import.json `
  --acknowledge-terms
```

Default selected base sequences:

`02,04,05,09,10,11,13`

To select explicitly:

```powershell
python -m spectratrack.vnext_qa import-mot17 `
  --dataset-root benchmarks/vnext/qa/public/MOT17 `
  --variant FRCNN `
  --sequences 02,04,05,09,10,11,13 `
  --output benchmarks/vnext/qa/imports/mot17-public.jsonl `
  --manifest benchmarks/vnext/qa/imports/mot17-public.import.json `
  --acknowledge-terms
```

## MOT17 conversion policy

Canonical scoring target:

- class 1 pedestrian, mark > 0 -> scored `person`;
- stable ID becomes `MOT17-XX:<original-id>`;
- classes 2, 7, 8, 12 (person-on-vehicle, static person, distractor, reflection) ->
  `person, ignore=true`, including the MOT GT rows whose consider/ignore flag is `0`;
- zero-marked pedestrian GT -> omitted from scored GT;
- unrelated classes -> omitted;
- original 1-based source frame -> `source_frame`;
- canonical evaluator frame -> source frame minus 1;
- source sequence/FPS are preserved;
- official full-body bbox is converted from 1-based xywh to 0-based xyxy **without clipping**;
- target-like boxes may extend beyond the image, but boxes with no image intersection are omitted because they cannot be observed or matched by the canonical evaluator;
- visibility is preserved numerically and only converted into explicitly named numeric
  visibility bins;
- semantic scene tags are added only where the official MOT17 sequence description supports them.

Current objective metadata tags:

- MOT17-04: `night_dark`, `high_angle_cctv`;
- MOT17-05: `camera_motion`;
- MOT17-10: `night_dark`, `camera_motion`;
- MOT17-11: `camera_motion`;
- MOT17-09 records `low_angle` only;
- MOT17-02 / MOT17-13 receive no inferred condition tags.

No automatic `partial_occlusion` or `heavy_occlusion` label is invented from visibility.

## Validate

```powershell
python -m spectratrack.vnext_qa validate-corpus `
  --video-root benchmarks/vnext/qa/public/MOT17 `
  --golden-ground-truth benchmarks/vnext/qa/imports/mot17-public.jsonl `
  --coverage-profile public-dataset `
  --dataset-import benchmarks/vnext/qa/imports/mot17-public.import.json `
  --output benchmarks/vnext/qa/imports/mot17-public.validation.json
```

## Freeze

```powershell
python -m spectratrack.vnext_qa freeze-corpus `
  --video-root benchmarks/vnext/qa/public/MOT17 `
  --golden-ground-truth benchmarks/vnext/qa/imports/mot17-public.jsonl `
  --coverage-profile public-dataset `
  --dataset-import benchmarks/vnext/qa/imports/mot17-public.import.json `
  --revision mot17-public-r1 `
  --reviewer "MOTChallenge official GT" `
  --confirm-public-dataset-terms `
  --output benchmarks/vnext/qa/imports/mot17-public.manifest.json
```

Freeze re-hashes every source file listed by the importer. Moving files is fine only if the
relative layout under `--video-root` remains the same; changing source bytes invalidates the
import.

Important: after conversion, scoring uses SpectraTrack's canonical A5 matcher/ignore policy.
This is intentionally a common project evaluator, **not** a claim of bit-for-bit equivalence with
the official MOTChallenge TrackEval implementation.

MOT17 is appropriate for:

- tracking recall;
- ID switches;
- fragmentation;
- uninterrupted track length;
- recovery;
- temporal bbox stability;
- pedestrian detection metrics.

---

# 2. CrowdHuman validation

Use **validation**, not train, for the A5 detector benchmark.

## Download/layout

The official CrowdHuman page provides Google Drive/Baidu links for both
`CrowdHuman_val.zip` and `annotation_val.odgt`.

One reproducible Google Drive option:

```powershell
py -m pip install gdown

mkdir benchmarks\vnext\qa\public\CrowdHuman
cd benchmarks\vnext\qa\public\CrowdHuman

py -m gdown 18jFI789CoHTppQ7vmRSFEdnGaSQZ4YzO -O CrowdHuman_val.zip
py -m gdown 10WIRwu8ju8GRLuCkZ_vT6hnNxs5ptwoL -O annotation_val.odgt

Expand-Archive .\CrowdHuman_val.zip -DestinationPath .
cd ..\..\..\..\..
```

After extraction, identify the directory containing the validation JPG files. Archive layout can
vary; common examples are `Images` or a nested CrowdHuman validation directory.

Example desired layout:

```text
benchmarks/vnext/qa/public/CrowdHuman/
  annotation_val.odgt
  Images/
    <ID>.jpg
    ...
```

## Full-body policy — default

A5 default:

`--bbox-kind full`

This uses CrowdHuman `fbox`.

Reason: CrowdHuman explicitly provides separate visible-body and full-body annotations/tasks.
For SpectraTrack's primary whole-person CCTV evaluation, the frozen public detector corpus uses
the full-body target.

Visible-body `vbox` remains available for a separate diagnostic corpus, but **must not** be
mixed into the same frozen revision as `fbox`.

## Import full-body validation GT

```powershell
python -m spectratrack.vnext_qa import-crowdhuman `
  --dataset-root benchmarks/vnext/qa/public/CrowdHuman `
  --annotations annotation_val.odgt `
  --images-dir Images `
  --bbox-kind full `
  --output benchmarks/vnext/qa/imports/crowdhuman-val-fbox.jsonl `
  --manifest benchmarks/vnext/qa/imports/crowdhuman-val-fbox.import.json `
  --acknowledge-terms
```

If the JPEGs extracted elsewhere, change only `--images-dir`.

Conversion policy:

- `tag=person` and not ignored -> scored `person`;
- `tag=mask` OR `extra.ignore=1` -> canonical `ignore=true`;
- selected `fbox`/ `vbox` xywh -> xyxy without clipping;
- source boxes may extend outside image bounds, but target-like boxes with no image intersection are omitted because canonical QA cannot observe or match them;
- original fbox/vbox/box_id/occ remain in source provenance;
- non-ignored integer `extra.occ` is exposed literally as an attribute such as
  `crowdhuman_occ_1`, so existing attribute metrics can report it without assigning an
  undocumented semantic meaning;
- no semantic CCTV condition tags are guessed;
- no stable canonical object ID is created;
- therefore CrowdHuman contributes **no tracking/ID metrics**.

## Validate/freeze

```powershell
python -m spectratrack.vnext_qa validate-corpus `
  --video-root benchmarks/vnext/qa/public/CrowdHuman `
  --golden-ground-truth benchmarks/vnext/qa/imports/crowdhuman-val-fbox.jsonl `
  --coverage-profile public-dataset `
  --dataset-import benchmarks/vnext/qa/imports/crowdhuman-val-fbox.import.json `
  --output benchmarks/vnext/qa/imports/crowdhuman-val-fbox.validation.json
```

```powershell
python -m spectratrack.vnext_qa freeze-corpus `
  --video-root benchmarks/vnext/qa/public/CrowdHuman `
  --golden-ground-truth benchmarks/vnext/qa/imports/crowdhuman-val-fbox.jsonl `
  --coverage-profile public-dataset `
  --dataset-import benchmarks/vnext/qa/imports/crowdhuman-val-fbox.import.json `
  --revision crowdhuman-val-fbox-r1 `
  --reviewer "CrowdHuman official validation GT" `
  --confirm-public-dataset-terms `
  --output benchmarks/vnext/qa/imports/crowdhuman-val-fbox.manifest.json
```

As with MOT17, resulting metrics use the common SpectraTrack A5 evaluator. They should not be
described as official CrowdHuman benchmark numbers unless the official evaluation protocol is run
separately.

Visible-body diagnostic variant:

```powershell
python -m spectratrack.vnext_qa import-crowdhuman `
  --dataset-root benchmarks/vnext/qa/public/CrowdHuman `
  --annotations annotation_val.odgt `
  --images-dir Images `
  --bbox-kind visible `
  --output benchmarks/vnext/qa/imports/crowdhuman-val-vbox.jsonl `
  --manifest benchmarks/vnext/qa/imports/crowdhuman-val-vbox.import.json `
  --acknowledge-terms
```

Freeze it under a different revision such as `crowdhuman-val-vbox-r1`.

---

# 3. Import provenance

Each `*.import.json` records:

- dataset name/version;
- split;
- applicable terms/license metadata;
- terms acknowledgement;
- selected MOT sequences or CrowdHuman validation split;
- conversion settings;
- bbox policy;
- importer exact source commit;
- every annotation/source-image path + SHA-256 + byte length;
- aggregate source-file provenance hash;
- converted JSONL SHA-256;
- deterministic import-manifest SHA-256.

`validate-corpus --dataset-import ...` re-hashes the source bytes before accepting the import.

`freeze-corpus` then embeds the validated import manifest in the normal A5 frozen corpus manifest.
There is no alternate evaluator after conversion.

---

# 4. Hybrid evaluation suite

Recommended evidence suite:

1. `mot17-public-r1`
   - tracking / IDs / temporal boxes / moving-camera and crowded pedestrian cases;
2. `crowdhuman-val-fbox-r1`
   - dense person detection / crowd / occlusion / full-body detection;
3. `cctv-golden-r1`
   - small private human-confirmed user CCTV holdout for actual SpectraTrack domain evidence.

Keep these as **separate frozen corpus revisions/hashes**. Their semantics differ and should not be
hidden inside one aggregate score.

A1-A4 should report the relevant rows on the same frozen public/private revisions.

Public datasets substantially reduce manual annotation work, but they do not prove performance on
the user's CCTV. Product GO still requires the small private holdout.
# 5. DanceTrack Round-2 held-out association corpus

Official source:

- `https://github.com/DanceTrack/DanceTrack`
- official dataset link from that repository: `https://huggingface.co/datasets/noahcao/dancetrack`

Terms recorded by A5:

- annotations: CC BY 4.0;
- dataset images/videos: non-commercial research only;
- code: MIT.

Raw DanceTrack media remains under the gitignored `benchmarks/vnext/qa/public/` area.

Expected official layout under the selected dataset root:

```text
dancetrack/
  train/
    dancetrack0001/
      img1/
      gt/gt.txt
      seqinfo.ini
  val/
    ...
  test/
    ...
```

Round-2 held-out import uses `val`:

```powershell
python -m spectratrack.vnext_qa import-dancetrack `
  --dataset-root benchmarks/vnext/qa/public/DanceTrack/dancetrack `
  --split val `
  --output benchmarks/vnext/qa/imports/dancetrack-val.jsonl `
  --manifest benchmarks/vnext/qa/imports/dancetrack-val.import.json `
  --acknowledge-terms
```

Conversion contract:

- reuses the existing generic MOT `seqinfo.ini`, MOT-row parser, 1-based xywh -> 0-based xyxy conversion, and source hashing;
- stable canonical ID: `DanceTrack:<sequence>:<original-id>`;
- original source frame remains in `source_frame`;
- exact sequence remains in `source_sequence`;
- official trailing fields must be exactly `1,1,1`; they are treated only as DanceTrack format constants;
- no MOT17 class IDs, distractor categories, visibility bins, or semantic scene tags are inherited;
- every sequence verifies seqinfo frame count and dimensions against the real images;
- public test GT is not fabricated and `test` is rejected by the importer.

Validate:

```powershell
python -m spectratrack.vnext_qa validate-corpus `
  --video-root benchmarks/vnext/qa/public/DanceTrack/dancetrack `
  --golden-ground-truth benchmarks/vnext/qa/imports/dancetrack-val.jsonl `
  --coverage-profile public-dataset `
  --dataset-import benchmarks/vnext/qa/imports/dancetrack-val.import.json `
  --output benchmarks/vnext/qa/imports/dancetrack-val.validation.json
```

Freeze only after the real local dataset import/validation succeeds:

```powershell
python -m spectratrack.vnext_qa freeze-corpus `
  --video-root benchmarks/vnext/qa/public/DanceTrack/dancetrack `
  --golden-ground-truth benchmarks/vnext/qa/imports/dancetrack-val.jsonl `
  --coverage-profile public-dataset `
  --dataset-import benchmarks/vnext/qa/imports/dancetrack-val.import.json `
  --revision dancetrack-public-r1 `
  --reviewer "DanceTrack official validation GT" `
  --confirm-public-dataset-terms `
  --output benchmarks/vnext/qa/imports/dancetrack-public-r1.manifest.json
```

`dancetrack-public-r1` is a held-out A2 association/ID gate. Do not retune the Round-2 ambiguity candidate from its validation metrics.

# 6. NightOwls Round-2 night corpus

Use only the official NightOwls distribution and official SDK:

- download page: `https://www.nightowls-dataset.org/download/`
- official SDK: `https://gitlab.com/vgg/nightowlsapi`
- validation images: `https://thor.robots.ox.ac.uk/~vgg/data/nightowls/python/nightowls_validation.zip`
- validation annotations: `https://thor.robots.ox.ac.uk/~vgg/data/nightowls/python/nightowls_validation.json`

Terms:

- non-commercial research / teaching / personal experimentation;
- citation required;
- dataset or modified versions may not be redistributed.

Example local setup from `pc/`:

```powershell
mkdir benchmarks\vnext\qa\public\NightOwls
cd benchmarks\vnext\qa\public\NightOwls
curl.exe -L https://thor.robots.ox.ac.uk/~vgg/data/nightowls/python/nightowls_validation.zip -o nightowls_validation.zip
curl.exe -L https://thor.robots.ox.ac.uk/~vgg/data/nightowls/python/nightowls_validation.json -o nightowls_validation.json
Expand-Archive .\nightowls_validation.zip -DestinationPath .
git clone https://gitlab.com/vgg/nightowlsapi.git nightowlsapi
cd ..\..\..\..
```

Point `--images-dir` at the extracted directory containing the official validation PNG files.

Full validation import:

```powershell
python -m spectratrack.vnext_qa import-nightowls `
  --dataset-root benchmarks/vnext/qa/public/NightOwls `
  --annotations nightowls_validation.json `
  --images-dir nightowls_validation `
  --sdk-dir nightowlsapi `
  --output benchmarks/vnext/qa/imports/nightowls-val.jsonl `
  --manifest benchmarks/vnext/qa/imports/nightowls-val.import.json `
  --acknowledge-terms
```

Fail-closed official semantics:

- category id 1 must be named `pedestrian` and is the only scored target class;
- pedestrian `ignore` is preserved;
- the official category named `ignore` becomes a canonical ignore region;
- bicycledriver, motorbikedriver and other non-pedestrian categories are never silently relabeled as ordinary pedestrians;
- official `occluded`, `difficult`, `pose_id`, `truncated`, tracking id, recording id, timestamp and daytime metadata are preserved where present;
- `night_dark` is added only when the official image metadata literally says `daytime=night`;
- a stable canonical tracking ID is emitted only when all scored pedestrians have valid official `tracking_id` values and at least one repeated trajectory is observed;
- canonical track ID is namespaced by recording: `NightOwls:<recording>:<tracking_id>`;
- image ordering inside a recording is determined only from official timestamp/image metadata.

The importer also hashes the local official SDK files used to establish the COCO/evaluation contract.

## Optional deterministic Round-2 validation slice

If full YOLO11x/960 validation is too expensive, freeze a slice **before any candidate result**:

```powershell
python -m spectratrack.vnext_qa import-nightowls `
  --dataset-root benchmarks/vnext/qa/public/NightOwls `
  --annotations nightowls_validation.json `
  --images-dir nightowls_validation `
  --sdk-dir nightowlsapi `
  --slice-frames 5000 `
  --slice-seed spectratrack-round2-nightowls-v1 `
  --logical-prefix golden/public/nightowls-val-slice `
  --output benchmarks/vnext/qa/imports/nightowls-val-slice.jsonl `
  --manifest benchmarks/vnext/qa/imports/nightowls-val-slice.import.json `
  --acknowledge-terms
```

Slice selection uses only official image/annotation metadata. It deterministically covers observed:

- positive and negative/background images;
- bbox-height strata;
- occlusion;
- difficulty;
- pose;
- official daytime metadata.

No detector/candidate output participates in selection. Selected image IDs and their deterministic hash are recorded in the import manifest.

Because a sparse slice breaks temporal continuity, `tracking_supported=false` for the slice even if the full official validation JSON contains valid tracking IDs. A1/A3 may share the frozen slice; A2 temporal conclusions require the full sequence import.

Validate full or sliced import with the same canonical command:

```powershell
python -m spectratrack.vnext_qa validate-corpus `
  --video-root benchmarks/vnext/qa/public/NightOwls `
  --golden-ground-truth benchmarks/vnext/qa/imports/nightowls-val.jsonl `
  --coverage-profile public-dataset `
  --dataset-import benchmarks/vnext/qa/imports/nightowls-val.import.json `
  --output benchmarks/vnext/qa/imports/nightowls-val.validation.json
```

Full validation freeze:

```powershell
python -m spectratrack.vnext_qa freeze-corpus `
  --video-root benchmarks/vnext/qa/public/NightOwls `
  --golden-ground-truth benchmarks/vnext/qa/imports/nightowls-val.jsonl `
  --coverage-profile public-dataset `
  --dataset-import benchmarks/vnext/qa/imports/nightowls-val.import.json `
  --revision nightowls-public-r1 `
  --reviewer "NightOwls official validation GT" `
  --confirm-public-dataset-terms `
  --output benchmarks/vnext/qa/imports/nightowls-public-r1.manifest.json
```

If a slice is used instead, its frozen revision must explicitly include `slice` and must never be reported as the full `nightowls-public-r1` result.

# 7. Secondary Round-2 corpora

LLVIP:

- optional secondary evidence only;
- visible/RGB side only;
- detection/enhancement only;
- documented detection boxes do not justify invented stable tracking identities.

KAIST:

- fallback only;
- visible/RGB side only for SpectraTrack inference;
- thermal/LWIR is not SpectraTrack production input;
- do not activate it unless NightOwls/LLVIP evidence is insufficient.

# 8. Private CCTV timing

Do not ask for review of the existing 46-frame draft now.

After public finalists are selected, reduce the private gate to roughly 10–15 hardest standalone frames plus 3–5 temporal episodes. Human action should be CONFIRM / FIX / REJECT. AI-only private annotations remain DRAFT and must never be frozen as `cctv-golden-r1`.

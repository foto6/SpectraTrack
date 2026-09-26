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
  --output benchmarks/vnext/qa/mot17-public.jsonl `
  --manifest benchmarks/vnext/qa/mot17-public.import.json `
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
  --output benchmarks/vnext/qa/mot17-public.jsonl `
  --manifest benchmarks/vnext/qa/mot17-public.import.json `
  --acknowledge-terms
```

## MOT17 conversion policy

Canonical scoring target:

- class 1 pedestrian, mark > 0 -> scored `person`;
- stable ID becomes `MOT17-XX:<original-id>`;
- classes 2, 7, 8, 12 (person-on-vehicle, static person, distractor, reflection) ->
  `person, ignore=true`;
- zero-marked pedestrian GT -> omitted from scored GT;
- unrelated classes -> omitted;
- original 1-based source frame -> `source_frame`;
- canonical evaluator frame -> source frame minus 1;
- source sequence/FPS are preserved;
- official full-body bbox is converted from 1-based xywh to 0-based xyxy **without clipping**;
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
  --golden-ground-truth benchmarks/vnext/qa/mot17-public.jsonl `
  --coverage-profile public-dataset `
  --dataset-import benchmarks/vnext/qa/mot17-public.import.json `
  --output benchmarks/vnext/qa/mot17-public.validation.json
```

## Freeze

```powershell
python -m spectratrack.vnext_qa freeze-corpus `
  --video-root benchmarks/vnext/qa/public/MOT17 `
  --golden-ground-truth benchmarks/vnext/qa/mot17-public.jsonl `
  --coverage-profile public-dataset `
  --dataset-import benchmarks/vnext/qa/mot17-public.import.json `
  --revision mot17-public-r1 `
  --reviewer "MOTChallenge official GT" `
  --confirm-public-dataset-terms `
  --output benchmarks/vnext/qa/mot17-public.manifest.json
```

Freeze re-hashes every source file listed by the importer. Moving files is fine only if the
relative layout under `--video-root` remains the same; changing source bytes invalidates the
import.

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
  --output benchmarks/vnext/qa/crowdhuman-val-fbox.jsonl `
  --manifest benchmarks/vnext/qa/crowdhuman-val-fbox.import.json `
  --acknowledge-terms
```

If the JPEGs extracted elsewhere, change only `--images-dir`.

Conversion policy:

- `tag=person` and not ignored -> scored `person`;
- `tag=mask` OR `extra.ignore=1` -> canonical `ignore=true`;
- selected `fbox`/ `vbox` xywh -> xyxy without clipping;
- original fbox/vbox/box_id/occ remain in source provenance;
- no semantic CCTV condition tags are guessed;
- no stable canonical object ID is created;
- therefore CrowdHuman contributes **no tracking/ID metrics**.

## Validate/freeze

```powershell
python -m spectratrack.vnext_qa validate-corpus `
  --video-root benchmarks/vnext/qa/public/CrowdHuman `
  --golden-ground-truth benchmarks/vnext/qa/crowdhuman-val-fbox.jsonl `
  --coverage-profile public-dataset `
  --dataset-import benchmarks/vnext/qa/crowdhuman-val-fbox.import.json `
  --output benchmarks/vnext/qa/crowdhuman-val-fbox.validation.json
```

```powershell
python -m spectratrack.vnext_qa freeze-corpus `
  --video-root benchmarks/vnext/qa/public/CrowdHuman `
  --golden-ground-truth benchmarks/vnext/qa/crowdhuman-val-fbox.jsonl `
  --coverage-profile public-dataset `
  --dataset-import benchmarks/vnext/qa/crowdhuman-val-fbox.import.json `
  --revision crowdhuman-val-fbox-r1 `
  --reviewer "CrowdHuman official validation GT" `
  --confirm-public-dataset-terms `
  --output benchmarks/vnext/qa/crowdhuman-val-fbox.manifest.json
```

Visible-body diagnostic variant:

```powershell
python -m spectratrack.vnext_qa import-crowdhuman `
  --dataset-root benchmarks/vnext/qa/public/CrowdHuman `
  --annotations annotation_val.odgt `
  --images-dir Images `
  --bbox-kind visible `
  --output benchmarks/vnext/qa/crowdhuman-val-vbox.jsonl `
  --manifest benchmarks/vnext/qa/crowdhuman-val-vbox.import.json `
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

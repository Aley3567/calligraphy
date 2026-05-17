# Algorithm Audit and Next Stage

## Current Decision

Training is paused. The current project should stop blind model iteration and move from engineering trial runs to structure-aware algorithm design.

The current main checkpoint remains:

```text
U-Net L1 baseline best.pt
baseline best val_l1 = 0.12746000836292903
```

The following runs are not promoted:

```text
Pix2Pix:
  final val_l1 = 0.19153394765324064
  reason: local texture realism damaged global character structure

U-Net resume:
  final val_l1 = 0.12922164450089138
  reason: did not beat baseline best and continued the dark/thick stroke failure mode
```

## Task Re-definition

This project is not ordinary paired image-to-image translation. The correct first-stage task is:

```text
weakly paired, single-style, structure-preserving Chinese calligraphy glyph generation
```

Current baseline mapping:

```text
x = standard font rendered content glyph
y = target calligraphy glyph
y_hat = f_theta(x)
```

This is only an engineering baseline. It is incomplete because it does not explicitly model:

```text
character content
glyph topology
stroke skeleton
internal holes / white-space
style
ink density
layout and bbox variation
scan noise
```

The real objective is not just visual style transfer. The hard constraint is:

```text
generated character must remain readable and structurally correct
```

Style is a secondary constraint. Local brush realism cannot be allowed to damage glyph topology.

## Why U-Net Baseline Works

U-Net works as a baseline because skip connections preserve spatial information from the input glyph:

```text
y_hat = Dec(Enc(x), skip(x))
```

This helps carry basic stroke layout, contour, and character identity from input to output.

Its limit is the L1 objective:

```text
L1(theta) = mean(|f_theta(x_i) - y_i|)
```

L1 optimizes pixel closeness, not character correctness. It does not know that:

```text
日 must keep an inner hole
田 must keep internal separation
国 must not become a black block
complex components must stay readable
```

This explains the observed failure mode:

```text
black blobs
thicker strokes
internal white-space swallowed
complex characters losing detail
```

## Why Pix2Pix Failed

Pix2Pix optimizes:

```text
G* = arg min_G max_D L_cGAN(G, D) + lambda * L_L1(G)
```

PatchGAN focuses on local patch realism. It may reward:

```text
brush-like edges
ink texture
dark local strokes
local calligraphy appearance
```

But it does not directly enforce:

```text
character identity
global topology
component relation
internal holes
stroke count
readability
```

For Chinese calligraphy, local realism can conflict with global structure. The current Pix2Pix result shows this conflict clearly, so plain Pix2Pix is not the main route.

Pix2Pix can only be reconsidered later as a weak texture refinement term after structure is already stable.

## Why Resume Training Is Not Promoted

Resume training started from the original U-Net baseline best checkpoint:

```text
start: epoch36 best.pt
continue: epoch37 -> epoch86
lr: 5e-5
```

Final result:

```text
train_l1 = 0.07452412473419567
val_l1 = 0.12922164450089138
```

It did not beat:

```text
baseline best val_l1 = 0.12746000836292903
```

It also continued the same visual risk:

```text
dark strokes
thicker strokes
internal details swallowed
```

Conclusion:

```text
More U-Net L1 training is not the solution.
```

## Main Failure Modes

The current core failures are:

```text
black ink too heavy
strokes too thick
internal white-space swallowed
complex glyph details disappear
high-risk characters become unreadable
val_l1 does not reflect readability
GAN improves local style while damaging structure
```

High-risk characters:

```text
日 月 田 国 民 夜 耀 翔 龟 鬱 齋 寝 恢 投 影 制 物
```

## Next Valid Algorithm Direction

The next main route should be:

```text
structure-aware U-Net
```

Not:

```text
plain U-Net resume
plain Pix2Pix
plain diffusion
FontDiffuser retry
refiner stacking
selector-only patching
```

Recommended model direction:

```text
input:
  content glyph x
  content mask M_x
  content skeleton S_x
  distance transform DT_x
  hole / white-space map H_x

output:
  mask head m_hat
  ink rendering head v_hat

final:
  y_hat = m_hat * v_hat
```

This separates:

```text
structure generation
ink rendering
```

The current single grayscale output entangles both, which is one cause of black blobs.

## Candidate Loss

A valid next loss should be closer to:

```text
L_total =
  lambda_gray    * L_gray
  + lambda_mask  * L_mask
  + lambda_edge  * L_edge
  + lambda_skel  * L_skeleton
  + lambda_density * L_density
  + lambda_hole  * L_hole
  + lambda_bbox  * L_bbox
```

Do not enable adversarial loss at first.

Suggested initial direction:

```text
lambda_gray = 1.0
lambda_mask = 1.0
lambda_edge = 0.2
lambda_skel = 0.5
lambda_density = 0.2
lambda_hole = 1.0
lambda_bbox = 0.05
lambda_adv = 0
```

Important: weights must be normalized by observed loss scale before serious training.

## Loss Roles

```text
L_gray:
  learns target calligraphy grayscale and ink tone

L_mask:
  learns cleaned target foreground shape

L_edge:
  improves boundary clarity

L_skeleton:
  protects content structure and reduces missing strokes

L_density:
  prevents over-dark, over-thick local ink blobs

L_hole:
  protects internal white-space for characters such as 日 田 回 国

L_bbox:
  prevents excessive scale, drift, and crop mismatch
```

Skeleton should not be forced as exact pixel alignment. It should be tolerance-band based:

```text
content skeleton points must have generated ink nearby
generated ink must not drift too far outside content structure
```

This protects readability without forcing the output to look exactly like printed font.

## Data Preprocessing Pipeline

Next training is not allowed until preprocessing is upgraded.

Required pipeline:

```text
raw gif
-> grayscale decode
-> polarity normalization: ink=1, background=0
-> background normalization
-> foreground bbox detection
-> aspect-ratio preserving crop and padding
-> 256x256 resize
-> cleaned gray target
-> clean mask
-> skeleton
-> distance transform
-> edge map
-> hole map from content glyph
-> metadata record
```

Required metadata:

```text
char
writer/style
source path
original size
bbox
scale
padding
ink ratio
local density summary
component / hole summary
filter flags
```

Bad samples must be flagged or filtered:

```text
too dark
too light
cropped
dirty background
lost internal holes
extreme bbox
suspected wrong glyph / variant conflict
```

## Evaluation Protocol

No model can be promoted using only:

```text
val_l1
preview image
subjective looks okay
```

Fixed test characters must include:

```text
simple:
  一 二 三 人 大 口

box:
  日 田 回 国

left-right:
  明 湖 海 翔 耀

top-bottom:
  空 夢 意

enclosure / semi-enclosure:
  闹 建 延 開

complex:
  龟 鬱 齋 鹤

high-risk:
  民 夜 寝 恢 投 影 制 物
```

Each board must separate:

```text
seen characters
unseen characters
high-risk characters
structure-type groups
```

Automatic metrics:

```text
val_l1
ink ratio
local ink density
bbox center/scale
edge sharpness
skeleton recall
skeleton precision
hole preservation
distance-transform similarity
component / topology proxy
```

Manual rating dimensions:

```text
readability
missing / extra strokes
structure correctness
internal white-space preservation
style consistency
ink darkness control
complex character readability
```

## Promotion Rule

A model can only replace baseline if:

```text
no high-risk character becomes unreadable
hole preservation improves or stays stable
local density does not worsen
skeleton recall and precision do not trade off badly
unseen characters are not worse than baseline
manual readability is not worse than baseline
style improvement does not damage structure
```

Hard failure:

```text
val_l1 improves but readability worsens
style improves but structure worsens
local density gets darker than baseline
box characters lose internal holes
complex characters become blobs
```

## Allowed Next Experiments

At most three branches are allowed. No branch should run full-scale before smoke validation.

### Branch 1: Structure-Ink Decoupled U-Net

Change:

```text
single grayscale output -> mask head + ink head
```

Loss:

```text
gray + mask + edge + density + hole + bbox
```

No skeleton yet. No GAN.

Goal:

```text
reduce black blobs and preserve internal white-space
```

### Branch 2: Add Skeleton Tolerance Loss

Only after Branch 1 is stable.

Change:

```text
add content skeleton tolerance loss
```

Goal:

```text
reduce missing strokes and complex-character collapse
```

### Branch 3: Weak Texture Refinement

Only after Branch 2 passes structure tests.

Change:

```text
add very weak adversarial / feature matching term
```

Goal:

```text
improve edge and brush texture without damaging structure
```

## Forbidden Work

Do not do:

```text
continue adding epochs to U-Net L1
run plain Pix2Pix again
start diffusion / FontDiffuser retry
stack refiner after refiner
judge by preview only
judge by val_l1 only
start full training before fixed evaluation exists
mix multiple styles before single-style structure is solved
mix English / digits / punctuation into Chinese model
```

## Current Project Status

Frozen promoted artifact:

```text
baseline_unet_l1_zhaomengfu_256_100ep/checkpoints/best.pt
```

Rejected or not promoted:

```text
Pix2Pix 100ep
U-Net resume epoch86
```

Next valid action:

```text
build preprocessing + evaluation + structure-aware loss infrastructure
```

Not:

```text
train another model immediately
```


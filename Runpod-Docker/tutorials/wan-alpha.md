# Wan-Alpha Tutorial

Use this quick reference to capture the key settings and workflow from [the companion YouTube tutorial](https://youtu.be/MbA1iUBHQzY).

## Install Unzip (ubuntu)

```bash
apt update
apt install unzip
```

## ffmpeg commands

To assemble a .mov file:

```bash
ffmpeg -framerate 24 -pattern_type glob -i 'img_*.png' \
  -vf format=rgba \
  -c:v prores_ks -profile:v 4444 -pix_fmt yuva444p10le -alpha_bits 16 \
  wan_alpha_transparent.mov
```

If your video has a gray background after running the above command, run:

```bash
ffmpeg -i wan_alpha_transparent.mov -c:v prores_ks -profile:v 4444 \
       -pix_fmt yuva444p10le -alpha_bits 16 \
       fixed_wan_alpha_transparent.mov
```

To assemble a .webm file (for web assets):

```bash
ffmpeg -framerate 24 -pattern_type glob -i 'img_*.png' \
  -vf "format=rgba" \
  -c:v libvpx-vp9 -lossless 1 -pix_fmt yuva420p \
  wan_alpha_transparent.webm
```

## Prompts

#### For golden retriever:

This video has a transparent background. A golden retriever puppy with fine, dense fur sways in the wind, its tail gently wagging. Realistic style. Medium shot.

#### For blonde woman:

This video has a transparent background. An attractive blonde woman, dressed in a stylish urban business suit, smiles as a gentle breeze rustles her long, flowing blonde hair, sending strands flying. Her eyes are gentle and vibrant, and a contented smile plays on her lips. Realistic style. Medium shot. Slight upward angle.

#### For emoji motion graphic:

This video has a transparent background. A cheerful round yellow emoji face. The emoji winks one eye, then its cheeks softly blush pink while its mouth curls into a shy but happy smile. Clean modern digital art style. Close-up shot.

#### For dog motion graphic:

This video has a transparent background. A cheerful cute puppy wags its tail and then runs away. Clean modern digital art style. Close-up shot.

# Third-Party Notices

This repository contains or depends on third-party material. The top-level
Apache License 2.0 applies only to original SemComp pipeline code. It does not
replace the licenses listed below.

## Panda-70M

Upstream project: <https://github.com/snap-research/Panda-70M>

Affected files include the shot/event splitting implementation under
`splitting/` (excluding SemComp-specific documentation and launch scripts) and
SemComp modifications or orchestration derived from that implementation.

The local version adds configurable paths, a combined full-pipeline entry
point, checkpoint handling, and integration with Stage 6. Modified files are
therefore not identical to the upstream release.

The upstream notice is reproduced below as required:

> Copyright (c) 2024 Snap Inc. All rights reserved. This dataset and code is
> made available by Snap Inc. for non-commercial, research purposes only.
> Non-commercial means not primarily intended for or directed towards
> commercial advantage or monetary compensation. Research purposes mean
> solely for study, instruction, or non-commercial research, testing or
> validation.
>
> No commercial license, whether implied or otherwise, is granted in or to
> this dataset and code, unless you have entered into a separate agreement
> with Snap Inc. for such rights. This dataset and code is provided as-is,
> without warranty of any kind, express or implied, including any warranties
> of merchantability, title, fitness for a particular purpose,
> non-infringement, or that the code is free of defects, errors or viruses. In
> no event will Snap Inc. be liable for any damages or losses of any kind
> arising from this dataset and code or your use thereof. Any redistribution
> of this dataset and code must retain or reproduce the above copyright
> notice, conditions and disclaimer.

Full upstream license: <https://github.com/snap-research/Panda-70M/blob/main/LICENSE>

## ImageBind

Upstream project: <https://github.com/facebookresearch/ImageBind>

Affected files: `splitting/ImageBind/**`.

The local copy is adapted for the repository's import layout, configurable
checkpoint location, and Stage 6 integration. It is licensed under the
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
license (CC BY-NC-SA 4.0). It may be used and redistributed only under those
terms, including attribution, non-commercial use, modification disclosure,
and ShareAlike requirements.

License text: <https://github.com/facebookresearch/ImageBind/blob/main/LICENSE>

Original material: <https://github.com/facebookresearch/ImageBind>

No endorsement by Meta, FAIR, or the ImageBind authors is implied.

## OpenAI CLIP

The ImageBind tokenizer vocabulary and portions of helper code trace to
OpenAI CLIP: <https://github.com/openai/CLIP>.

MIT License

Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## PySceneDetect

PySceneDetect is used as an installed dependency and is licensed under the BSD
3-Clause License.

Upstream project and license:
<https://github.com/Breakthrough/PySceneDetect/blob/main/LICENSE>

Copyright (C) 2014, Brandon Castellano.

## Model weights and source media

Downloaded ImageBind weights, external VLM services, and source videos are not
distributed by this repository. Their separate terms still apply. Users are
responsible for reviewing those terms and ensuring that they have the rights
needed for their data and intended use.

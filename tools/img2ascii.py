#!/usr/bin/env python3
"""
Turn an image into the ASCII art block of dark_mode.svg / light_mode.svg.

The geometry below is not arbitrary: it mirrors the SVG layout exactly.  The art
lives at x=15 and the info column starts at x=470, so the art gets 455px.  A
monospace glyph advances 0.6em, so at font-size 8px a cell is 4.8 x 10 px and 96
columns take 460.8px -- the same footprint the old 48-column, 16px art had.

The two SVGs need opposite ramps.  On the light card the glyphs are dark on a
light background, so ink means shadow; on the dark card they are light on a dark
background, so ink means highlight.  Using one ramp for both, as the old art did,
leaves one of the two cards showing a negative.

Preview first, write second:

    python3 tools/img2ascii.py logos/avatar.jpg --preview /tmp/prev
    python3 tools/img2ascii.py logos/avatar.jpg --crop 200,400,1800,2400 --write
"""

import argparse
import os
import sys
from xml.dom import minidom

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- SVG geometry -----------------------------------------------------------
COLS = 96
ROWS = 50
ART_X = 15          # x of every ascii tspan
ART_Y0 = 30         # baseline of the first row
LINE_H = 10         # baseline step
FONT_PX = 8
CHAR_W = FONT_PX * 0.6
CELL_ASPECT = LINE_H / CHAR_W   # 2.083: how much taller than wide a cell is

MONO = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'

# '<', '>' and '&' are left out so the art never depends on XML escaping.
CANDIDATES = (" .'`^\",:;Il!i~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
              .replace('&', ''))

THEMES = {
    'dark':  {'file': 'dark_mode.svg',  'bg': '#161b22', 'fg': '#c9d1d9', 'invert': True},
    'light': {'file': 'light_mode.svg', 'bg': '#f6f8fa', 'fg': '#24292f', 'invert': False},
}


def measure_glyphs():
    """
    Render every candidate and measure how it behaves *as a tile*, since a flat
    region of the image becomes the same glyph repeated hundreds of times.

    Two things ruin that tiling.  Ink touching the left and right edges of the
    cell joins up with its neighbours into a continuous rule across the picture
    -- that is why the first attempt at this ramp turned flat areas into ruled
    paper with '_'.  And ink sitting far from the vertical middle ('^', ',')
    stacks into stripes between rows.  Both get scored down.
    """
    font = ImageFont.truetype(MONO, 48)
    w, h = int(48 * 0.6021) + 2, 56
    rows = np.arange(h)[:, None]
    out = {}
    for ch in dict.fromkeys(CANDIDATES):
        img = Image.new('L', (w, h), 0)
        ImageDraw.Draw(img).text((1, 0), ch, fill=255, font=font)
        a = np.asarray(img, dtype=np.float64) / 255.0
        cov = a.mean()
        if cov < 1e-6:                       # the space
            out[ch] = (cov, 0.0, 0.0)
            continue
        centroid = (a * rows).sum() / a.sum()
        off_center = abs(centroid - (h - 1) / 2) / (h / 2)
        edge = (a[:, :2].mean() + a[:, -2:].mean()) / (2 * cov)
        out[ch] = (cov, off_center, edge)
    return out


def build_ramp(size, forced=None):
    """
    Order characters by the ink they actually put on the page.

    Eyeballing a ramp like " .:-=+*#%@" assumes each glyph is darker than the
    last, which is only roughly true.  Measuring gives steps evenly spaced in
    real coverage, which is what keeps gradients smooth instead of banded --
    then, among the glyphs that hit a given step, prefer the one whose ink sits
    evenly in its cell so flat regions read as texture, not as ruled paper.
    """
    glyphs = measure_glyphs()
    if forced:
        ramp = list(forced)
        levels = np.array([glyphs[c][0] for c in ramp])
        return ramp, (levels - levels[0]) / (levels[-1] - levels[0])

    cov = {c: v[0] for c, v in glyphs.items()}
    lo, hi = cov[' '], max(cov.values())
    ramp, used = [], set()
    for i in range(size):
        target = lo + (hi - lo) * i / (size - 1)
        ch = min((c for c in glyphs if c not in used),
                 key=lambda c: (abs(cov[c] - target) / (hi - lo) * 3.0
                                + glyphs[c][1] * 0.6 + glyphs[c][2] * 0.8))
        used.add(ch)
        ramp.append(ch)
    ramp.sort(key=lambda c: cov[c])
    levels = np.array([cov[c] for c in ramp])
    return ramp, (levels - levels[0]) / (levels[-1] - levels[0])


def foreground_mask(img, tol):
    """
    Flood the plain backdrop away, starting from the border.

    A flat luminance threshold is not enough: a lit forehead can be as bright as
    the wall behind it.  What separates them is that the wall touches the edge of
    the frame and the forehead does not, so this keeps only the bright pixels
    reachable from the border, and leaves everything enclosed by the subject.
    """
    small = img.resize((512, round(512 * img.size[1] / img.size[0])), Image.LANCZOS)
    a = np.asarray(small, dtype=np.float64) / 255.0
    h, w = a.shape

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    # the backdrop is whatever sits within tol of the median brightness of the frame edge
    ref = np.median(a[border])
    bright = np.abs(a - ref) < tol

    bg = border & bright
    while True:                      # 4-neighbour propagation to a fixed point
        grown = bg.copy()
        grown[1:, :] |= bg[:-1, :]
        grown[:-1, :] |= bg[1:, :]
        grown[:, 1:] |= bg[:, :-1]
        grown[:, :-1] |= bg[:, 1:]
        grown &= bright
        if grown.sum() == bg.sum():
            break
        bg = grown

    fg = Image.fromarray(((~bg) * 255).astype(np.uint8))
    return fg.filter(ImageFilter.GaussianBlur(1)).resize(img.size, Image.BILINEAR)


def stretch(lum, fg, cut=(2, 98)):
    """
    Spread the subject over the full range, ignoring the backdrop.

    Stretching across the whole frame lets a bright backdrop own the top of the
    range and leaves the face squashed into a few middle steps of a 12-step ramp,
    which is most of what makes a converted photo look like grey mush.
    """
    sel = fg > 0.5
    if not sel.any():
        return lum
    lo, hi = np.percentile(lum[sel], cut)
    return np.clip((lum - lo) / (hi - lo), 0.0, 1.0) if hi > lo else lum


def load_image(path, crop, gamma, sharpen, fill, rmbg, local, flatten):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)      # iPhone photos carry rotation in EXIF
    img = img.convert('L')

    if crop:
        img = img.crop(tuple(int(v) for v in crop.split(',')))

    mask = foreground_mask(img, rmbg) if rmbg else Image.new('L', img.size, 255)
    img = ImageOps.autocontrast(img, cutoff=1)

    w, h = img.size
    if fill:
        # crop to the exact cell-corrected aspect of the box so the art fills it
        want = (COLS * CHAR_W) / (ROWS * LINE_H)
        cur = w / h
        if cur > want:
            new_w = int(h * want)
            box = ((w - new_w) // 2, 0, (w - new_w) // 2 + new_w, h)
        else:
            new_h = int(w / want)
            box = (0, (h - new_h) // 2, w, (h - new_h) // 2 + new_h)
        img, mask = img.crop(box), mask.crop(box)
        cols, rows = COLS, ROWS
    else:
        rows = round(COLS * (h / w) / CELL_ASPECT)
        if rows <= ROWS:
            cols = COLS
        else:
            rows, cols = ROWS, round(ROWS * CELL_ASPECT / (h / w))

    lum = np.asarray(img.resize((cols, rows), Image.LANCZOS), dtype=np.float64) / 255.0
    fg = np.asarray(mask.resize((cols, rows), Image.LANCZOS), dtype=np.float64) / 255.0

    # Stretch contrast across the subject alone. Stretching across the whole
    # frame lets a bright backdrop own the top of the range and leaves the face
    # squashed into a few middle steps of a 12-step ramp, which is most of what
    # makes a converted photo look like grey mush.
    lum = stretch(lum, fg)

    if local:
        # Split the picture into a blurred base and the detail on top of it, then
        # turn the base down and the detail up.
        #
        # This is the move that makes a face survive 12 grey levels. Untouched,
        # the huge dark mass of hair and the bright skin eat the whole ramp
        # between them: the hair fills in as one solid block and the glasses,
        # eyes and mouth vanish into flat highlight. Damping the large-scale
        # difference and boosting the small-scale one gives the features back and
        # lets the curls read as texture.
        #
        # It has to happen here, on the 96x50 grid. Sharpening the 2216px
        # original would only touch detail that the downscale to 96 columns
        # throws away.
        blur = np.asarray(Image.fromarray((lum * 255).astype(np.uint8))
                          .filter(ImageFilter.GaussianBlur(local)), dtype=np.float64) / 255.0
        mid = lum[fg > 0.5].mean() if (fg > 0.5).any() else 0.5
        lum = np.clip(mid + (blur - mid) * flatten + (lum - blur) * sharpen, 0.0, 1.0)
        lum = stretch(lum, fg)

    if gamma != 1.0:
        lum = lum ** gamma
    return lum, fg


def to_grid(lum, fg, ramp, levels, invert, dither, knockout):
    """Map luminance to characters and pad the result to exactly COLS x ROWS."""
    ink = lum.copy() if invert else 1.0 - lum
    if knockout is not None:
        # the backdrop is bright, so it is bright pixels that go blank -- in both
        # themes, otherwise the dark card renders a wall of glyphs with a
        # person-shaped hole punched out of it
        ink[lum > knockout] = 0.0
    ink *= fg                       # fades the art out over the removed backdrop

    rows, cols = ink.shape
    out = np.zeros((rows, cols), dtype=int)
    if dither:
        work = ink.copy()
        for y in range(rows):
            for x in range(cols):
                old = work[y, x]
                idx = int(np.argmin(np.abs(levels - old)))
                out[y, x] = idx
                err = old - levels[idx]
                if x + 1 < cols:
                    work[y, x + 1] += err * 7 / 16
                if y + 1 < rows:
                    if x > 0:
                        work[y + 1, x - 1] += err * 3 / 16
                    work[y + 1, x] += err * 5 / 16
                    if x + 1 < cols:
                        work[y + 1, x + 1] += err * 1 / 16
    else:
        out = np.abs(levels[None, None, :] - ink[:, :, None]).argmin(axis=2)

    lines = [''.join(ramp[i] for i in row) for row in out]

    pad_l = (COLS - cols) // 2
    lines = [' ' * pad_l + ln + ' ' * (COLS - cols - pad_l) for ln in lines]
    pad_t = (ROWS - rows) // 2
    blank = ' ' * COLS
    return [blank] * pad_t + lines + [blank] * (ROWS - rows - pad_t)


def render_preview(lines, theme, path, scale=4):
    """
    Approximate what the SVG will look like.  There is no SVG renderer on this
    box, so we redraw the grid with the same cell metrics the SVG uses.
    """
    cw, ch = CHAR_W * scale, LINE_H * scale
    img = Image.new('RGB', (int(COLS * cw), int(ROWS * ch)), theme['bg'])
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(MONO, FONT_PX * scale)
    for y, line in enumerate(lines):
        for x, chx in enumerate(line):
            if chx != ' ':
                draw.text((x * cw, y * ch), chx, fill=theme['fg'], font=font)
    img.save(path)
    return path


def ascii_text_element(svg):
    for node in svg.getElementsByTagName('text'):
        if node.getAttribute('class') == 'ascii':
            return node
    raise SystemExit('no <text class="ascii"> element found')


def write_svg(lines, theme):
    path = os.path.join(REPO, theme['file'])
    svg = minidom.parse(path)

    # the .ascii class exists on the element but has no rule; this is what
    # shrinks the glyphs to fit 96 columns in the same 455px band
    style = svg.getElementsByTagName('style')[0]
    css = style.firstChild.data
    if '.ascii' not in css:
        style.firstChild.data = css.replace(
            'text, tspan', '.ascii {font-size: %dpx;}\ntext, tspan' % FONT_PX)

    text = ascii_text_element(svg)
    while text.firstChild:
        text.removeChild(text.firstChild)
    for i, line in enumerate(lines):
        text.appendChild(svg.createTextNode('\n'))
        tspan = svg.createElement('tspan')
        tspan.setAttribute('x', str(ART_X))
        tspan.setAttribute('y', str(ART_Y0 + i * LINE_H))
        tspan.appendChild(svg.createTextNode(line))
        text.appendChild(tspan)
    text.appendChild(svg.createTextNode('\n'))

    with open(path, 'w', encoding='utf-8') as f:
        f.write(svg.toxml('utf-8').decode('utf-8'))
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('image')
    p.add_argument('--crop', help='L,T,R,B in source pixels, applied first')
    p.add_argument('--gamma', type=float, default=1.0,
                   help='<1 lifts the midtones, >1 deepens them (default 1.0)')
    p.add_argument('--local', type=float, default=0.0, metavar='CELLS',
                   help='radius of the local-contrast pass, in grid cells; off by '
                        'default because on a portrait it fights the dithering and '
                        'just adds noise')
    p.add_argument('--sharpen', type=float, default=2.0,
                   help='gain on local detail (glasses, eyes, curls)')
    p.add_argument('--flatten', type=float, default=0.45,
                   help='how much large-scale contrast to keep, 0-1; lower stops the '
                        'hair from filling in as one solid mass')
    p.add_argument('--ramp-size', type=int, default=12)
    p.add_argument('--ramp', help='force a ramp, lightest to densest, e.g. " .:*ox%%@"')
    p.add_argument('--dither', action='store_true',
                   help='Floyd-Steinberg over the ramp; smoother large gradients')
    p.add_argument('--knockout', type=float,
                   help='blank out pixels brighter than this (0-1), in both themes')
    p.add_argument('--rmbg', type=float, metavar='TOL',
                   help='drop the backdrop: bright pixels reachable from the frame '
                        'edge, within TOL of the edge brightness (try 0.15-0.3)')
    p.add_argument('--fill', action='store_true',
                   help='crop to fill the whole box instead of padding with spaces')
    p.add_argument('--preview', help='write <prefix>_dark.png and <prefix>_light.png')
    p.add_argument('--write', action='store_true', help='update the two SVGs')
    p.add_argument('--stdout', action='store_true', help='print the light-mode grid')
    args = p.parse_args()

    if not args.preview and not args.write and not args.stdout:
        p.error('nothing to do: pass --preview, --stdout or --write')

    ramp, levels = build_ramp(args.ramp_size, args.ramp)
    lum, fg = load_image(args.image, args.crop, args.gamma, args.sharpen, args.fill,
                         args.rmbg, args.local, args.flatten)
    print('ramp: %r' % ''.join(ramp), file=sys.stderr)
    print('sampled grid: %dx%d  foreground: %.0f%%'
          % (lum.shape[1], lum.shape[0], 100 * (fg > 0.5).mean()), file=sys.stderr)

    for name, theme in THEMES.items():
        lines = to_grid(lum, fg, ramp, levels, theme['invert'], args.dither, args.knockout)
        assert len(lines) == ROWS and all(len(l) == COLS for l in lines)
        if args.stdout and name == 'light':
            print('\n'.join(lines))
        if args.preview:
            # 1x is the size the card is actually viewed at, and the only honest
            # test of whether the portrait reads; 4x is for inspecting glyphs
            for scale in (1, 4):
                print('preview: %s' % render_preview(
                    lines, theme, '%s_%s_%dx.png' % (args.preview, name, scale), scale),
                    file=sys.stderr)
        if args.write:
            print('wrote: %s' % write_svg(lines, theme), file=sys.stderr)


if __name__ == '__main__':
    main()

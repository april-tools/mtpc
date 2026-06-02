import argparse
import pandas as pd

# Code translated from a matplotlib plot using Claude and adjusted

PALETTE = [
    ("tabBlue",   "31,119,180"),    # #1f77b4
    ("tabOrange", "255,127,14"),    # #ff7f0e
    ("tabGreen",  "44,160,44"),     # #2ca02c
    ("tabRed",    "214,39,40"),     # #d62728
    ("tabPurple", "148,103,189"),   # #9467bd
    ("tabBrown",  "140,86,75"),     # #8c564b
]


def generate_tikz(csv_path='mtpc2.csv', font='times'):
    """
    Reads the MTPC data and prints standalone TikZ/pgfplots code
    reproducing the matplotlib scatter plot with iso-throughput lines.

    Encoding:
      * colour       -> Model         (Okabe-Ito colourblind-safe palette)
      * ring count   -> LoRA Layers    (rings stack INWARD from the rim as a
                         thick black border; fixed outer diameter for all k)
      * inset number -> context length n ('8' / '16' printed at the centre
                         of every marker)

    Ring scheme (every marker has a thin black outer border):
      LoRA 0 : thin border, colour to centre.
      LoRA 1 : thin border fused with a black ring (reads as one thick
               border), colour to centre.
      LoRA 2 : thin border, white ring, black ring, colour to centre.

    One custom plot mark is declared per (LoRA level, model colour) pair,
    with the model colour baked into the core -- there is no public PGF
    primitive to read the active fill colour from inside a mark, so the
    colour must be fixed at declaration time. With a handful of models
    and LoRA levels this is only a dozen marks.

    font : 'times'    -> \\usepackage{times} (URW Nimbus Roman, Times-metric
                         clone; compile with pdflatex).
           'fontspec' -> fontspec with "Times New Roman"; requires
                         lualatex/xelatex and the font installed.
    """
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"LoRA Layers": "lora_layers"})
    df = df.sort_values("Model")

    # ---- Axis limits --------------------------------------------------
    # xmin is pulled out to 4.8 (the data starts at ~5.1) so the inset
    # iso-line value labels have an empty left margin to sit in without
    # shadowing any data points. xticks are pinned to the round 5.0,
    # 5.5, ... positions so the extended range adds no stray 4.8 tick.
    if csv_path == 'mtpc2.csv':
        xmin, xmax = 4.8, 7.7
        ymin, ymax = 0.027, 0.044
        xticks = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5]

        # ---- Iso-throughput constants -------------------------------------
        # Spaced by 5 and including 175 so the FF cluster (throughput ~175)
        # sits on a labelled line.
        cs = list(range(155, 206, 5))
    elif csv_path == "mtpc3-llama-argmax-nolora.csv":
        xmin, xmax = 2.5, 4.5
        ymin, ymax = 0.02, 0.044
        xticks = [2.5, 3.0, 3.5, 4.0, 4.5]

        cs = list(range(105, 150, 5))
    elif csv_path == "mtpc3-llama-sampling-nolora.csv":
        xmin, xmax = 1.5, 2.5
        ymin, ymax = 0.02, 0.044
        xticks = [1.5, 2.0, 2.5]

        cs = list(range(55, 80, 5))
    elif csv_path == "mtpc3-evabyte-sampling-nolora.csv":
        xmin, xmax = 5., 7.5
        ymin, ymax = 0.025, 0.044
        xticks = [5., 5.5, 6., 6.5, 7., 7.5]

        cs = list(range(160, 205, 5))
    elif csv_path == "mtpc3-evabyte-argmax-nolora.csv":
        xmin, xmax = 6.5, 9.0
        ymin, ymax = 0.025, 0.045
        xticks = [6.5, 7., 7.5, 8., 8.5, 9.]

        cs = list(range(205, 270, 10))
    else:
        raise ValueError(f"Unknown input file {csv_path} - check if range needs adapting")

    # ---- Encodings ----------------------------------------------------
    models = sorted(df["Model"].unique())
    lora_vals = sorted(int(v) for v in df["lora_layers"].unique())
    model_colour = {m: PALETTE[i % len(PALETTE)][0]
                    for i, m in enumerate(models)}

    # ---- Marker geometry ----------------------------------------------
    # Every marker is a model-colour disc with ONE border of fixed
    # thickness. The LoRA level is encoded by the border's darkness on a
    # full white -> mid-grey -> black ramp (LoRA 0 / 1 / 2). A thin dark
    # outline rings the whole marker so the white (LoRA 0) border still
    # reads with high contrast against the plot background.
    # R   : fixed outer radius shared by EVERY marker (footprint constant).
    # ol  : thin dark outline thickness (outermost).
    # bw  : LoRA-grey border thickness (just inside the outline).
    R = 6.5     # pt, fixed outer radius for ALL markers
    ol = 0.15    # pt, thin dark outline around the whole marker
    bw = 2.    # pt, LoRA-grey border thickness

    # Border shades per LoRA level: a full white -> black ramp for maximum
    # luminance separation. Index = LoRA k.
    BORDER_GREY = {
        0: ("loraBorder0", "255,255,255"),   # white
        1: ("loraBorder1", "128,128,128"),   # mid grey
        2: ("loraBorder2", "0,0,0"),         # black
    }

    # ---- Build the TikZ string ----------------------------------------
    lines = []
    A = lines.append

    A(r"\documentclass[tikz,border=3pt]{standalone}")
    A(r"\usepackage{pgfplots}")
    if font == 'fontspec':
        A(r"\usepackage{fontspec}")
        A(r"\setmainfont{Times New Roman}")
    else:
        A(r"\usepackage{times}")
    A(r"\pgfplotsset{compat=1.18}")

    # --- Named colours  -------------------------------------
    for name, rgb in PALETTE:
        A(f"\\definecolor{{{name}}}{{RGB}}{{{rgb}}}")
    # neutral grey core for the worked-example legend marker
    A(r"\definecolor{egcore}{RGB}{240,240,240}")
    # border greys encoding the LoRA level (light / medium / black)
    A(r"% --- LoRA-level border greys (light / medium / black) ---------")
    for k, (gname, grgb) in sorted(BORDER_GREY.items()):
        A(f"\\definecolor{{{gname}}}{{RGB}}{{{grgb}}}")

    # --- Custom plot marks: one per (LoRA level, model colour) ---------
    # Each mark is two filled discs: an outer border disc in the LoRA
    # grey, then a model-colour core disc inset by the border thickness.
    A(r"% --- custom marks: single grey border, model-colour core ------")

    # Legend swatches are drawn at a reduced scale so the legend box stays
    # compact; data markers use scale 1.0.
    LEG_SCALE = 0.72
    leg_mark_size = R * LEG_SCALE   # mark size for legend swatches

    def emit_mark_body(k, core_colour, inset_text=None, scale=1.0,
                       inset_font=r"\scriptsize"):
        """Emit the \\pgfdeclareplotmark body for a single-border marker.

        Three concentric filled discs, outermost first:
          1) a thin dark outline (radius R) so the marker silhouette --
             and especially the white LoRA-0 border -- always reads;
          2) the LoRA-level border (radius R-ol) on the white->black
             ramp, of thickness bw;
          3) the model-colour core (radius R-ol-bw).
        The inset digit, if any, sits on the model-colour core.
        'scale' shrinks every radius uniformly (used for legend swatches).
        """
        rr = R * scale
        oll = ol * scale
        bb = bw * scale
        border_colour = BORDER_GREY[k][0]
        # 1) thin dark outline disc
        A(r"  \pgfsetfillcolor{black}%")
        A(f"  \\pgfpathcircle{{\\pgfpointorigin}}{{{rr:.3f}pt}}%")
        A(r"  \pgfusepath{fill}%")
        # 2) LoRA-level border disc, inset by the outline thickness
        r_border = rr - oll
        A(f"  \\pgfsetfillcolor{{{border_colour}}}%")
        A(f"  \\pgfpathcircle{{\\pgfpointorigin}}{{{r_border:.3f}pt}}%")
        A(r"  \pgfusepath{fill}%")
        # 3) model-colour core disc, inset by outline + border thickness
        r_core = rr - oll - bb
        A(f"  \\pgfsetfillcolor{{{core_colour}}}%")
        A(f"  \\pgfpathcircle{{\\pgfpointorigin}}{{{r_core:.3f}pt}}%")
        A(r"  \pgfusepath{fill}%")
        # 4) optional inset digits, white, centred on the core
        if inset_text is not None:
            A(f"  \\pgftext[base,center]{{\\color{{white}}"
              f"{inset_font}\\bfseries\\raisebox{{-0.5ex}}{{{inset_text}}}}}%")

    # mark name for a (LoRA level k, colour name) pair
    def mark_name(k, cname):
        return f"mk{k}{cname}"

    # 'egcore' (neutral grey core) variants are used by the legend's
    # LoRA border-grey ramp, where the border is what should stand out.
    used_colours = sorted(set(model_colour.values())) + ["egcore"]
    for k in lora_vals:
        for cname in used_colours:
            # full-size data marker
            A(f"\\pgfdeclareplotmark{{{mark_name(k, cname)}}}{{%")
            emit_mark_body(k, cname)
            A(r"}")
            # reduced-size legend swatch
            A(f"\\pgfdeclareplotmark{{{mark_name(k, cname)}Lg}}{{%")
            emit_mark_body(k, cname, scale=LEG_SCALE)
            A(r"}")

    # --- Borderless model swatches for the legend ----------------------
    # The model legend row encodes colour only, so its swatches are plain
    # filled discs with no border (a 'plainLg' variant per model colour).
    A(r"% --- borderless legend swatches (colour only) -----------------")
    for cname in sorted(set(model_colour.values())):
        A(f"\\pgfdeclareplotmark{{plain{cname}Lg}}{{%")
        A(f"  \\pgfsetfillcolor{{{cname}}}%")
        A(f"  \\pgfpathcircle{{\\pgfpointorigin}}"
          f"{{{R * LEG_SCALE:.3f}pt}}%")
        A(r"  \pgfusepath{fill}%")
        A(r"}")

    # --- Inset-digit swatches for the legend ---------------------------
    # One swatch per distinct MTP window value: a dark-grey disc (no LoRA
    # border) with the white digit baked in, so the white inset font is
    # legible. Used by the 'MTP window size' legend row.
    n_values = sorted(int(v) for v in df["n"].unique())
    A(r"% --- inset-digit legend swatches (dark-grey background) -------")
    A(r"\definecolor{insetbg}{RGB}{90,90,90}")
    for nv in n_values:
        A(f"\\pgfdeclareplotmark{{inset{nv}Lg}}{{%")
        A(f"  \\pgfsetfillcolor{{insetbg}}%")
        A(f"  \\pgfpathcircle{{\\pgfpointorigin}}"
          f"{{{R * LEG_SCALE:.3f}pt}}%")
        A(r"  \pgfusepath{fill}%")
        A(f"  \\pgftext[base,center]{{\\color{{white}}"
          f"\\tiny\\bfseries\\raisebox{{-0.5ex}}{{{nv}}}}}%")
        A(r"}")

    # --- Inline-mark macro: draws a legend swatch inside running text --
    # \legmark{<markname>} renders one custom plot mark as an inline box,
    # so two model swatches+names can share a single legend row. The mark
    # size must be set explicitly (\pgfuseplotmark honours \pgfsetplotmark
    # size) and the mark is drawn at the origin of a small tikzpicture.
    A(r"% --- inline mark macro for packing two models per legend row --")
    A(f"\\pgfsetplotmarksize{{{leg_mark_size:.2f}pt}}")
    # \legmarkraw draws one custom mark inside a tikzpicture whose
    # bounding box is set explicitly to the marker's true extent (the
    # custom \pgfdeclareplotmark uses absolute pt circles that TikZ
    # cannot measure on its own, so without this the picture has zero
    # size and the disc overruns adjacent text).
    A(r"\newcommand{\legmarkraw}[1]{%")
    A(r"  \tikz[baseline=-0.7ex]{%")
    A(f"    \\pgfsetplotmarksize{{{leg_mark_size:.2f}pt}}%")
    A(f"    \\useasboundingbox (-{leg_mark_size:.2f}pt,-{leg_mark_size:.2f}pt) "
      f"rectangle ({leg_mark_size:.2f}pt,{leg_mark_size:.2f}pt);%")
    A(r"    \pgfpathmoveto{\pgfpointorigin}%")
    A(r"    \pgfuseplotmark{#1}}%")
    A(r"}")
    # \legmark reserves a fixed box so the mark never overlaps the text
    # that follows it. Box width = marker diameter + a little padding.
    mark_box_pt = 2 * leg_mark_size + 4.0
    A(f"\\newcommand{{\\legmark}}[1]{{%")
    A(f"  \\makebox[{mark_box_pt:.1f}pt][c]{{\\legmarkraw{{#1}}}}%")
    A(r"}")

    A(r"\begin{document}")
    A(r"\begin{tikzpicture}")
    A(r"\begin{axis}[")
    A(r"    width=10cm, height=8cm,")
    A(f"    xmin={xmin}, xmax={xmax},")
    A(f"    ymin={ymin}, ymax={ymax},")
    A(r"    xlabel={\Large Accepted tokens $\rightarrow$},")
    A(r"    ylabel={\Large $\leftarrow$ Latency (s)},")
    A(r"    xlabel style={font=\Large},")
    A(r"    ylabel style={font=\Large},")
    A(r"    scaled y ticks=false,")
    A(f"    xtick={{{','.join(f'{t}' for t in xticks)}}},")
    A(r"    ytick={0.028,0.032,0.036,0.040,0.044},")
    A(r"    y tick label style={/pgf/number format/fixed,"
      r" /pgf/number format/fixed zerofill,"
      r" /pgf/number format/precision=3},")
    A(r"    axis lines=box,")
    A(r"    clip=false,")
    A(r"    enlargelimits=false,")
    A(r"    legend style={font=\footnotesize, at={(0.025,0.975)},"
      r" anchor=north west, draw=black, fill=white, fill opacity=1,"
      r" text opacity=1, legend columns=1, row sep=2pt,"
      r" inner sep=3pt, /tikz/nodes={inner sep=1pt}},")
    A(r"    legend cell align=left,")
    A(r"]")

    # --- Iso-throughput dashed lines (y = x/c), clipped in data space --
    for c in cs:
        xa = max(xmin, c * ymin)
        xb = min(xmax, c * ymax)
        if xa < xb:
            A(f"  \\addplot[dashed, line width=0.4pt, black, opacity=0.5, "
              f"forget plot] coordinates "
              f"{{({xa:.6f},{xa / c:.6f}) ({xb:.6f},{xb / c:.6f})}};")

    # --- Label each iso-line with its own throughput value -------------
    # The line y = x/c crosses the plot box from lower-left to upper-
    # right. Steeper (high-c) lines leave through the right edge and are
    # labelled just outside it. Shallower (low-c) lines would leave
    # through the left edge; those are labelled in the empty margin
    # between the y-axis (x=xmin) and the first data (~x=5.1), so e.g.
    # 175 sits clearly beside the FF cluster without shadowing points.
    x_inset = xmin + 0.5 * (5.0 - xmin)     # middle of the left margin
    for c in cs:
        y_right = xmax / c
        if ymin <= y_right <= ymax:
            A(f"  \\node[anchor=west, font=\\small, xshift=6pt] "
              f"at (axis cs:{.99 * xmax},{1.01 * y_right:.6f}) {{{c}}};")
        else:
            y_inset = x_inset / c
            if ymin <= y_inset <= ymax:
                A(f"  \\node[font=\\small, fill=white, inner sep=1pt, "
                  f"rounded corners=1pt] "
                  f"at (axis cs:{x_inset:.6f},{y_inset:.6f}) {{{c}}};")

    # --- "Throughput" label, above the iso-line value column -----------
    # Throughput = accepted tokens / latency, so it rises toward the
    # steep (high-c) iso-lines, i.e. downward on this plot -- hence the
    # downward arrow. Shifted right so it sits over the value labels.
    A(r"  \node[anchor=south east, font=\large, xshift=20pt] "
      r"at (rel axis cs:1,1) {\large Throughput (tok/s) $\downarrow$};")

    # --- Scatter points: ring markers ----------------------------------
    for (model, lora), g in df.groupby(["Model", "lora_layers"]):
        cname = model_colour[model]
        k = int(lora)
        coords = " ".join(
            f"({r.acceptance_rate},{r.latency})" for r in g.itertuples()
        )
        A(f"  % Model={model}, LoRA Layers={k}")
        A(f"  \\addplot[only marks, mark={mark_name(k, cname)}, "
          f"mark size={R:.2f}pt, forget plot] "
          f"coordinates {{{coords}}};")

    # --- Inset numbers: context length n, centred on each marker -------
    # Uses pgfplots 'nodes near coords': the label is read from the third
    # coordinate (point meta) of an invisible \addplot drawn last, so it
    # always paints on top of the markers. White digits on dark fills.
    A(r"  % --- inset context-length numbers (n) ---")
    label_coords = " ".join(
        f"({r.acceptance_rate},{r.latency}) [{int(r.n)}]"
        for r in df.itertuples()
    )
    A(r"  \addplot[only marks, mark=none, point meta=explicit symbolic,")
    A(r"    nodes near coords, every node near coord/.append style={")
    A(r"      font=\scriptsize\bfseries, text=white, anchor=center,")
    A(r"      inner sep=0pt}, forget plot]")
    A(f"    coordinates {{{label_coords}}};")

    # --- Legend: single column, two models packed per row -------------
    # Rows 1-2 each hold two models. The swatches are drawn inline in the
    # label text with \legmark, and the row's \addlegendimage is empty,
    # so pgfplots does not reserve a wide marker column. This avoids the
    # wasted space that 'legend columns=2' creates (it would force both
    # columns to the width of the worked-example row).
    #
    # Each model name is set in a fixed-width left-aligned \makebox so the
    # two columns line up vertically regardless of name length. The box
    # width is measured from the longest model name.
    A(r"  % --- legend: models, two per row ---")
    # Each model name sits in a fixed-width left-aligned box so the two
    # columns align. Width is a generous em value (scales with the font)
    # rather than a measured length, which proved fragile across the
    # legend's font cascade.
    name_box_em = max(len(m) for m in models) * 0.62 + 0.6
    pairs = [models[i:i + 2] for i in range(0, len(models), 2)]
    for pair in pairs:
        cells = []
        for m in pair:
            cm = model_colour[m]
            cells.append(
                f"\\legmark{{plain{cm}Lg}}"
                f"\\makebox[{name_box_em:.2f}em][l]{{{m}}}"
            )
        label = "".join(cells)
        A(r"  \addlegendimage{empty legend}")
        A(f"  \\addlegendentry{{{label}}}")

    # --- Worked-example rows -------------------------------------------
    # Row 3: the LoRA border-grey ramp -- three swatches labelled 0,1,2.
    # Row 4: the MTP window sizes -- one dark-grey digit swatch per n.
    A(r"  % --- legend: LoRA border-grey ramp ---")
    ramp_cells = []
    for k in sorted(lora_vals):
        ramp_cells.append(
            f"\\legmark{{{mark_name(k, 'egcore')}Lg}}"
            f"\\makebox[1.0em][l]{{{k}}}"
        )
    ramp = r"\hspace{0.3em}".join(ramp_cells)
    A(r"  \addlegendimage{empty legend}")
    A(f"  \\addlegendentry{{LoRA layers:\\hspace{{0.4em}}{ramp}}}")

    A(r"  % --- legend: MTP window sizes ---")
    win_cells = [f"\\legmark{{inset{nv}Lg}}" for nv in n_values]
    wins = r"\hspace{0.3em}".join(win_cells)
    A(r"  \addlegendimage{empty legend}")
    A(f"  \\addlegendentry{{MTP window size:\\hspace{{0.4em}}{wins}}}")

    A(r"\end{axis}")
    A(r"\end{tikzpicture}")
    A(r"\end{document}")

    tikz_code = "\n".join(lines)
    print(tikz_code)
    return tikz_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate TikZ/pgfplots code for the MTPC scatter plot."
    )
    parser.add_argument(
        "--csv", default="mtpc2.csv",
        help="Path to the input CSV file (default: mtpc2.csv).",
    )
    args = parser.parse_args()
    generate_tikz(args.csv)

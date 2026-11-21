# MS-DBSCAN Studio

Segments a triangle mesh of any size with MS-DBSCAN v2-fuse and the coarse-to-fine
pyramid, and replays how the segmentation is built.

```
python app/msdbscan_studio.py [mesh.obj|.ply]
```

## What it does

1. **Coarse level**: the published preprocessing (90 % reduction, 100 Laplacian
   iterations), capped at 6000 vertices for very large meshes. The map from every
   original vertex to its coarse vertex is kept.
2. **Parameters**: either chosen per mesh (the highest normal-space silhouette among
   60 configurations × merge off/on; no ground truth) or set by hand.
3. **MS-DBSCAN v2-fuse** on the coarse level: n-ring ∩ ball in the fused normal space
   (normal × (1 + λ|curvature × mean edge|)), canonical DBSCAN semantics, unassigned
   vertices joined to a neighbour.
4. **Pyramid** to the full mesh: lift labels through the collapse map, clear a band
   around boundaries, reassign band vertices to the segment with the closest local
   proxy normal, refill pieces cut off from their segment.

The computation runs in a worker thread; the window stays responsive.

## Layout

The window follows the original MS-DBSCAN application (`legacy/original_app/app/GUI_remastered_Qt.py`):
a *Basic* tab with a control panel of fixed width on the left and the 3D view on the right, a *Detail*
tab with one panel per space (below), and a *Help* tab with this text. The panel's regions have a red border (*3D files list*, *Surface
segmentation*, *Replay stages*, *Export*); the parameter groups inside *Surface segmentation* have a
blue one (*Settings*, *Thresholds*). Pick a folder with **Choose a Folder**, select a mesh and press
**Preview** (or double-click it), then **Compute**. With *Parameters = Auto (silhouette)* the thresholds
and hole filling are chosen per mesh; with *Manual* the values under *Thresholds* are used
(**Load default** restores 0.16 / 3 / 0.3).

Every region has an arrow before its title: click the arrow (or the title) to fold the region to one
line, and again to unfold it. *Thresholds* is folded while the parameters are automatic and unfolds
when *Manual* is chosen; *Replay stages* and *Export* unfold when a result arrives.

## Detail tab

The original application showed the normal space and the curvature in separate views but not the
n-ring. The *Detail* tab shows every space MS-DBSCAN combines, one panel each, on the coarse level of
the last result (where the clustering runs) and with its thresholds:

| Panel | Shows |
|---|---|
| Normal space | the unit normal of every vertex, as a point on the unit sphere |
| Curvature | c = maximum principal curvature × mean edge length, on the mesh (red convex, blue concave) |
| Fused space | every normal stretched to length 1 + λ·\|c\| (the space the eps ball is taken in) |
| n-ring | the vertices at most n edges away on the mesh |
| N(i) | the neighbourhood DBSCAN counts: n-ring ∩ fused ball; a vertex is core when it has at least 4 |
| Result | the segmentation of the coarse level, before it is carried to the full mesh |

Before a pick, the sphere panels colour the points by segment, the *n-ring* panel shows how many
vertices each n-ring holds, and the *N(i)* panel shows the size of N(i), with non-core vertices in grey
(typically the creases). **Click a vertex** in any panel, on the mesh (the vertex of the face under the
cursor nearest the click) or on a sphere (the point nearest the cursor; neighbouring vertices often share
a pixel there), to see its
neighbourhood in every space: red = the vertex (with its eps ball on the spheres), green = n-ring ∩ ball,
orange = n-ring only, blue = ball only (same orientation but too far along the surface), grey = the
rest. The line above the panels gives the counts. Dragging turns the view; the four mesh panels turn
together, and so do the two spheres. **Clear** returns to the maps.

The quantities come from `src/msdbscan/spaces.py`, which uses v2's own functions;
the self-test checks that its core vertices are the ones the clustering
recorded.

## Parameters

### Settings

| Control | What it does |
|---|---|
| **Parameters** | *Auto (silhouette)*: on the coarse level, segments the mesh with each of the 60 configurations of the search grid (Eps normal ∈ {0.08, 0.12, 0.16, 0.24, 0.32} × Eps n-ring ∈ {3, 5, 8} × λ ∈ {0, 0.1, 0.3, 1}), once without and once with hole filling, and keeps the segmentation whose segments are best separated in normal space: the area-weighted silhouette of the face normals, each segment represented by its mean normal. No ground truth is used. The chosen values are printed under the progress bar, and the first replay stage shows every candidate. *Manual (thresholds)*: uses the values under *Thresholds* and the *Hole filling* choice. |
| **Hole filling** | Merges small pieces after clustering (Algorithm 3 of the paper): every connected piece of a segment with fewer than T coarse vertices takes the label most frequent just outside it. T = max(min(20, ⌊0.025·V⌋), 4), with V the number of vertices of the coarse level. This is the rule of the released tool; the paper states max(⌊0.025·V⌋, 4), without the cap of 20. In Auto mode the search decides and the control is greyed out. |
| **To full resolution** | How the coarse labels reach the original mesh. *coarse only*: no transfer, the result is shown on the coarse level. *lift*: each original vertex takes the label of the coarse vertex it was collapsed into; boundaries keep the coarse zigzag. *band 1* (default): lift, then recompute the vertices within 1 coarse edge length of a boundary (each takes the segment whose local mean normal, over up to 16 nearby vertices that kept their label, is closest to its own normal) and refill pieces cut off from their segment. *band 2*: the same with a band of 2 edge lengths. *two steps*: the band-1 refinement first on an intermediate level keeping 30 % of the faces, then on the full mesh. |
| **Coarse level cap** | Clustering and the Auto search run on a coarse copy of the mesh, made by the published preprocessing: keep 10 % of the vertices, then 100 Laplacian smoothing iterations. When 10 % is more than the cap, the mesh is reduced to the cap instead; when 10 % would be under 100 vertices, the mesh is used as it is. A higher cap keeps smaller features but makes every configuration slower, and on a dense coarse level clusters chain across rounded edges. The default 6000 changes none of the 3D Puzzles fragments (largest coarse level: 5591 vertices). |
| **Show edge** | Display only. Draws the boundaries between segments as pink lines, 5 pixels wide, as the original application did (its `highlight_edge`): every mesh edge whose faces do not all carry the same label, on the replay frame shown (any two labels, so during the clustering also the border between a cluster and the not-yet-reached part) and on the *Result* panel of the Detail tab. The face labels are the majority vote of the three corners; the original application used the released `counter_sort`, which has a bug (it compares a label with its own count), so its face labels, and with them its lines, differ from these on some faces. The faces of each mesh edge are listed once per level by the worker thread (0.4 s on a 277 638-vertex scan), and a frame then costs one comparison per edge (8 ms on that scan), so switching it on or playing the replay does not pause; the original's vertex-by-vertex search would take about 76 minutes on that scan (estimated). |
| **Segment colours** | Display only. *Distinct hues* (default): every segment id has a fixed hue, far from the hues of the ids next to it, so neighbouring segments rarely look alike. *Jet (original app)*: the original application's colours, segment ids from blue (smallest) to red (largest id shown). Segments with close ids get close colours, and a segment can change colour when the largest id disappears (for example in *Merge small regions*). With jet, core elements are magenta in the replay, since jet has its own yellow. |
| **Map colours** | Display only, Detail tab: the colour map of the curvature panel and of the n-ring and N(i) panels. *Coolwarm, viridis* (default): curvature on a two-sided map, light grey at 0, red convex, blue concave; counts on viridis, whose brightness rises steadily. *Turbo*: a rainbow like jet, made to change brightness more smoothly. *Jet (original app)*: the original application's map; its brightness rises and falls, so a smooth change can show bands that are not in the data. The range is the same for all three: ± the 98th percentile of \|c\| (the original application used the 8th to 95th percentile of κ). |

### Thresholds (Manual only)

| Control | What it does |
|---|---|
| **Eps normal** | Radius in normal space. Two vertices can be neighbours when the distance between their unit normals is at most Eps normal. The distance is the chord between the two unit vectors, 2·sin(θ/2) for an angle θ between the normals: 0.08 ≈ 4.6°, 0.12 ≈ 6.9°, 0.16 ≈ 9.2°, 0.24 ≈ 13.8°, 0.32 ≈ 18.4°. A larger value joins surfaces that bend more: fewer, larger segments. With λ > 0 the vectors are longer than 1 and the same radius also bounds the curvature difference (see below). |
| **Eps n-ring** | Radius on the surface: only vertices at most n edges away on the coarse level are candidates. The neighbourhood of a vertex is (n-ring) ∩ (normal ball). A vertex with at least 4 neighbours (MinPts = 5, itself included) is a core vertex; clusters grow from core vertex to core vertex, and the remaining vertices join the core neighbour with the closest normal (DBSCAN). A larger n makes more vertices core and lets clusters cross narrow features. |
| **Curvature fuse λ** | Weight of curvature. Before the Eps normal test, each unit normal is stretched to length r = 1 + λ·\|κ·ē\|, with κ the maximum principal curvature at the vertex (VTK) and ē the mean edge length of the coarse level, which makes κ·ē independent of the object's size. λ = 0 is the normal-only space. κ·ē is small inside facets and large on creases, so a small λ mainly separates creases and a large λ also splits curved facets. |
| **Load default** | Sets 0.16 / 3 / 0.3, values inside the Auto grid. They are a starting point, not values selected on data. |

## Curvature separate and curvature fuse

The original application offers three feature types: *Normal only*, *Curvature fuse* and *Curvature
separate*. Both curvature types add curvature to the normal-space test; they differ in how a
difference in orientation and a difference in curvature are combined.

**Curvature separate** (feature type 3 of the original application, the method of the paper): two
thresholds and two tests, both of which must pass:

    ‖n_i − n_j‖ ≤ Eps normal   and   |κ_i − κ_j| ≤ Eps curvature

The accepted pairs form a rectangle in the plane (normal difference, curvature difference): a pair
with almost equal normals is still rejected when the curvatures differ by more than Eps curvature,
and the other way round. The curvature is signed, so a convex and a concave crease of the same
sharpness differ. In the paper κ is in 1/length units, so the same Eps curvature means a different
thing on a small and on a large object (the experiments' variant *v2-c* multiplies κ by the mean edge
length to remove this).

**Curvature fuse** (this application): one threshold and one test, on stretched normals:

    ‖r_i·n_i − r_j·n_j‖ ≤ Eps normal,   r = 1 + λ·|κ·ē|

Both differences add up in one distance, and λ sets the rate of exchange between them. When two
normals are parallel the distance is λ·| |κ_i·ē| − |κ_j·ē| |, so with Eps normal 0.16 and λ = 0.3 the two
vertices stay neighbours up to a curvature difference of about 0.53 (in units of 1/ē). When the
curvatures are equal the distance is r times the normal distance, so the normal test is stricter on
strongly curved parts. There is no second threshold to tune, the sign of κ is ignored, and λ = 0
switches curvature off, so the Auto search can also decide that curvature does not help on a mesh.

The original application's *Curvature fuse* (feature type 2) uses another formula: each normal is
scaled by 1 + 0.1·(tanh(0.05·κ) − 0.5), with κ in 1/length units, so its length stays between 0.85
and 1.05. The constants 0.1, 0.05 and 0.5 are not given in the paper, and the curvature threshold is
not used. This application offers only its own curvature fuse; curvature separate is evaluated in the
experiments (`src/msdbscan/baselines.py`, *v2-c*).

## Replay

After a run the stage list shows every recorded step:

| Stage | Level | What changes |
|---|---|---|
| Parameter search | coarse | one frame per candidate; caption gives the parameters and silhouette |
| Core elements | coarse | gold (magenta with jet) = core (enough neighbours within eps) |
| Cluster expansion | coarse | clusters grow breadth-first from their densest core element |
| Border elements | coarse | non-core vertices join the closest-normal core neighbour |
| Unassigned elements join a neighbour | coarse | breadth-first rounds |
| Merge small regions | coarse | released hole-filling rule |
| Lift / Clear band / Refine band / Reconnect | each finer level | grey = being recomputed |

Colours: light grey = not reached, dark grey = unassigned, gold = core not yet drawn (magenta with
*Segment colours* set to jet).
The line under the progress bar counts the segments and, apart from them, the vertices left unassigned.
On the full-resolution Amazone scan (277 638 vertices; `evaluation_dataset/Real_CH` holds a 27 826-vertex version) these are the 11 small pieces of the scan that touch no other part
(3 to 127 vertices each, 321 in all): "4 segments, 321 vertices unassigned".
Play / pause, step, drag the slider, change the speed, or click a stage to jump to it.
"Replay to GIF…" writes the animation (needs `imageio`); "Save labels…" writes an
`.npz` with `V`, `F`, `face_labels`, `vertex_labels` and the parameters.

## Check without a display

```
mkdir selftest_out
python app\msdbscan_studio.py --selftest evaluation_dataset\SIGGRAPH_2006\brick_part06.obj selftest_out\brick_part06.png
```

On the 55 896-vertex fragment: 6 segments, 7–13 s including the parameter search,
10 stages, 168 frames. It checks that the count shown leaves the unassigned
label out, then the Detail tab (core vertices identical to the clustering; a real click on a
vertex visible from the camera picks that vertex, and a click on its point in the fused-space panel picks a point drawn under the cursor)
and the colour settings (switching to jet redraws both tabs, with the same colour per segment in each, and switching back
restores the first colours), and *Show edge* (the lines drawn are exactly the mesh edges whose faces carry different
labels, counted again by brute force, none remain when it is switched off, and the tables of the faces of each edge
come from the worker thread, so the window never builds one). The self-test also writes
`<out>_window.png` (the whole window), `<out>_detail.png` (the Detail tab), `<out>_jet.png` (the Detail tab in jet),
`<out>_edges.png` (the window with the segment boundaries) and `<out>_selftest.txt` (the result, for builds without a console).

## Release builds (PyInstaller)

PyInstaller does not cross-compile, so each release is built on its own OS. Both use
`app/msdbscan_studio.spec` (one-folder build) and `app/build_release.py`, which builds, runs the
self-test above with the frozen executable, and archives the folder.

| OS | Command | Output in `release/` |
|---|---|---|
| Windows (Python 3.9) | `python app\build_release.py` | `MSDBSCAN-Studio-<version>-windows-x64.zip` |
| Linux (here WSL Ubuntu 22.04) | `bash app/build_linux.sh` | `MSDBSCAN-Studio-<version>-linux-x64.tar.gz` |

`build_linux.sh` makes a virtual environment outside the repository with the pinned versions of
`app/requirements-build.txt` (those of the Windows environment) and builds on the Linux file system.
Beside each archive: a `.sha256`, `build-info-<os>.json` (library versions, self-test result) and
`selftest-<os>/`. The Linux build needs glibc 2.35 or newer (Ubuntu 22.04, Debian 12 and later).

The algorithm code is in `src/msdbscan`; the app adds no clustering logic of its own.

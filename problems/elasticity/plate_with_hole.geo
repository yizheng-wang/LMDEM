SetFactory("OpenCASCADE");

// --------------------
// Parameters
// --------------------
L  = 1.0;      // plate size: [0,L]x[0,L]
cx = 0.5;      // hole center x
cy = 0.5;      // hole center y
r  = 0.2;      // hole radius

lc_outer = 0.02;   // mesh size on outer boundary
lc_hole  = 0.01;   // mesh size on hole boundary

// --------------------
// Geometry (Boolean cut)
// --------------------
Rectangle(1) = {0, 0, 0, L, L};
Disk(2)      = {cx, cy, 0, r, r};
BooleanDifference(3) = { Surface{1}; Delete; }{ Surface{2}; Delete; };

// --------------------
// Identify boundary curves
// --------------------
// Outer boundary curves are typically 4 lines; hole boundary is a circle/arc curve loop.
// We'll get boundaries by querying the surface boundary.
outer[] = Boundary{ Surface{3}; };

// `outer[]` includes both the outer square edges and the hole curve(s).
// We separate them using a bounding-box filter: hole curves are near (cx,cy).
hole[] = Curve In BoundingBox {cx-r-1e-6, cy-r-1e-6, -1, cx+r+1e-6, cy+r+1e-6, 1};
sq[]   = Curve In BoundingBox {-1e-6, -1e-6, -1, L+1e-6, L+1e-6, 1};

// remove hole curves from square set (sq includes hole too sometimes depending on kernel)
// safer: define Gamma_u as left edge, Gamma_t as right edge (example).
left[]  = Curve In BoundingBox {-1e-6, -1e-6, -1,  1e-6, L+1e-6, 1};
right[] = Curve In BoundingBox {L-1e-6, -1e-6, -1, L+1e-6, L+1e-6, 1};

// --------------------
// Mesh sizes (optional but recommended)
// --------------------
MeshSize{ PointsOf{ Curve{sq[]}; } } = lc_outer;
MeshSize{ PointsOf{ Curve{hole[]}; } } = lc_hole;

// --------------------
// Physical groups (names must match your code)
// --------------------
Physical Surface("Omega") = {3};

// Example BC setting:
// - Gamma_u: left edge (Dirichlet)
// - Gamma_t: right edge (traction)
Physical Curve("Gamma_u") = {left[]};
Physical Curve("Gamma_t") = {right[]};

// If you want traction on TOP instead, use:
// top[] = Curve In BoundingBox {-1e-6, L-1e-6, -1, L+1e-6, L+1e-6, 1};
// Physical Curve("Gamma_t") = {top[]};

// (Optional) you can also make hole boundary a traction group, e.g. pressure:
// Physical Curve("Gamma_t") = {right[], hole[]};

// --------------------
// Meshing options
// --------------------
Mesh.Algorithm = 6;   // Frontal-Delaunay for 2D (often good)
Mesh 2;

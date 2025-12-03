# CraterInfo.msg
# A compact message describing a detected crater in the rover frame

float32 x            # crater centroid x (meters, rover frame)
float32 y            # crater centroid y (meters, rover frame)
float32 radius       # crater radius (meters)
float32 confidence   # detection confidence (0.0 - 1.0)
float32 inlier_ratio # fraction of crater circumference inliers used to fit
builtin_interfaces/Time stamp  # time of detection
string id_guess      # optional: heuristic map id guess (empty if none)

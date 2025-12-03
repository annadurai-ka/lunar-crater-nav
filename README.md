# **Lunar Crater Navigation - Team 11**

This repository contains the ROS2 workspace for our EECE 5550 Mobile Robotics final project:
**Crater-based localization for lunar rover navigation using depth perception and a particle filter.**

The system includes:

* **crater_perception** - Detects craters from depth point clouds
* **crater_msgs** - Custom ROS2 message (`CraterInfo.msg`)
* **nav_pf** - Particle Filter localization using crater detections + odometry (just for reference test - simple code)
* **Synthetic point cloud generator** for testing
* **Crater visualization tools** for RViz

---

## **Workspace Build**

```bash
cd ~/lunar_nav_ws
colcon build
source install/setup.bash
```

---

## **Packages**

### **1. crater_perception/**

* Subscribes: `/camera/depth/points`
* Publishes:

  * `/crater_detections` (PoseArray)
  * `/crater_info` (CraterInfo.msg)
  * `/crater_markers` (RViz visualization)

Run detector:

```bash
ros2 run crater_perception crater_detector
```

Synthetic test:

```bash
ros2 run crater_perception test_publisher
```

---

### **2. crater_msgs/**

Contains the custom ROS2 message:

```
CraterInfo.msg
├─ float32 x
├─ float32 y
├─ float32 radius
├─ float32 confidence
├─ float32 inlier_ratio
├─ builtin_interfaces/Time stamp
└─ string id_guess
```

---

### **3. nav_pf/**

Particle Filter for crater-based localization.

Subscribes:

* `/odom`
* `/crater_info`

Publishes:

* `/pf_pose` (PoseStamped)

Run:

```bash
ros2 run nav_pf pf_localizer
```

---

## **Testing Workflow**

### **Synthetic testing (no Gazebo):**

Terminal 1 - point cloud generator:

```bash
ros2 run crater_perception test_publisher
```

Terminal 2 - crater detector:

```bash
ros2 run crater_perception crater_detector
```

Terminal 3 - PF localization:

```bash
ros2 run nav_pf pf_localizer
```

Terminal 4 - view outputs:

```bash
ros2 topic echo /pf_pose
```

RViz:

* Fixed Frame: `base_link`
* Add `/camera/depth/points`
* Add `/crater_markers`
* Add `/pf_pose`

---

## **Map File**

The PF uses an orbital crater map in CSV format:

```
x,y,radius
10,5,5
-5,-8,7
```

Modify `map_craters.csv` as needed per simulation.

---

## **Dependencies**

* ROS2 Humble
* OpenCV
* NumPy
* PCL / point cloud utilities
* Colcon build tools

---

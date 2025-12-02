# 1. Create virtual bot

```bash
dts duckiebot virtual create --type duckiebot --configuration DB21J vquarck
```
```bash
dts duckiebot virtual start vquarck
```

# 2. Attach and run the matrix

```bash
dts matrix run --standalone --embedded --map loop
```

In another terminal, attach to the matrix:

```bash
dts matrix attach vquarck map_0/vehicle_0
```

# 3. Build

```bash
# build once (or when you change code)
dts devel build -f
```
# 4. Run
run SLAM launcher against vquarck

```bash
dts devel run -R vquarck
```
It's launchers/default.sh by default, or specify another launcher with -L

```bash
dts devel run -R vquarck -L <your-launcher-name>
```

# 5. VNC

In another tab/terminal, build the VNC image and run it:
```bash
# dts gui --vnc vquarck <- will build the default image from `dt-gui-tools`
dts devel build --file Dockerfile.vnc
```


TODO change sandbox to our map with landmarks
<!-- 
rostopic list
rostopic echo /vquarck/slam_pose
rostopic echo /vquarck/slam_landmarks 
rostopic echo /vquarck/slam_path
-->

# Template: template-ros

This template provides a boilerplate repository
for developing ROS-based software in Duckietown.

**NOTE:** If you want to develop software that does not use
ROS, check out [this template](https://github.com/duckietown/template-basic).


## How to use it

### 1. Fork this repository

Use the fork button in the top-right corner of the github page to fork this template repository.


### 2. Create a new repository

Create a new repository on github.com while
specifying the newly forked template repository as
a template for your new repository.


### 3. Define dependencies

List the dependencies in the files `dependencies-apt.txt` and
`dependencies-py3.txt` (apt packages and pip packages respectively).


### 4. Place your code

Place your code in the directory `/packages/` of
your new repository.


### 5. Setup launchers

The directory `/launchers` can contain as many launchers (launching scripts)
as you want. A default launcher called `default.sh` must always be present.

If you create an executable script (i.e., a file with a valid shebang statement)
a launcher will be created for it. For example, the script file 
`/launchers/my-launcher.sh` will be available inside the Docker image as the binary
`dt-launcher-my-launcher`.

When launching a new container, you can simply provide `dt-launcher-my-launcher` as
command.
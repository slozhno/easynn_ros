cmake_minimum_required(VERSION 2.8.12)
project(easynn_ros)

add_compile_options(-std=c++11)

# Get the information about this package's buildtime dependencies
find_package(catkin REQUIRED
  COMPONENTS message_generation std_msgs sensor_msgs)

# Declare the message files to be built
add_message_files(FILES
	ObjectBox.msg
)

# Actually generate the language-specific message and service files


# Declare that this catkin package's runtime dependencies


###########
## Build ##
###########
# main node

catkin_python_setup()

generate_messages(DEPENDENCIES std_msgs)

catkin_package(
 CATKIN_DEPENDS message_runtime std_msgs
)

# ENTER CORRECT PYTHON SCRIPT NAME!!!!
catkin_install_python(PROGRAMS src/imageSubscriber.py src/main.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})


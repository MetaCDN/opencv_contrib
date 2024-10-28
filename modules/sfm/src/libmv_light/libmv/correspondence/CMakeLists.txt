# define the source files
SET(CORRESPONDENCE_SRC feature_matching.cc
                       matches.cc
                       nRobustViewMatching.cc)

# define the header files (make the headers appear in IDEs.)
FILE(GLOB CORRESPONDENCE_HDRS *.h)

ADD_LIBRARY(opencv.sfm.correspondence STATIC ${CORRESPONDENCE_SRC} ${CORRESPONDENCE_HDRS})

<<<<<<< HEAD
TARGET_LINK_LIBRARIES(correspondence LINK_PRIVATE ${GLOG_LIBRARIES} multiview)
=======
ocv_target_link_libraries(opencv.sfm.correspondence LINK_PRIVATE ${GLOG_LIBRARIES} opencv.sfm.multiview opencv_imgcodecs)
>>>>>>> 80f1ca2442982ed518076cd88cf08c71155b30f6
IF(TARGET Eigen3::Eigen)
  TARGET_LINK_LIBRARIES(opencv.sfm.correspondence LINK_PUBLIC Eigen3::Eigen)
ENDIF()


LIBMV_INSTALL_LIB(opencv.sfm.correspondence)

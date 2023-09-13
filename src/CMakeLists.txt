set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

# Add source to this project's executable.
file(GLOB_RECURSE ENGINE_SOURCE_FILES
    "${PROJECT_SOURCE_DIR}/src/engine/*.cpp"
    "${PROJECT_SOURCE_DIR}/src/engine/*.h"
) 
file(GLOB_RECURSE GAME_SOURCE_FILES
    "${PROJECT_SOURCE_DIR}/src/game/*.cpp"
    "${PROJECT_SOURCE_DIR}/src/game/*.h"
)

add_executable(
    Risk
    main.cpp
    ${ENGINE_SOURCE_FILES}
    ${GAME_SOURCE_FILES}
)

### LINKING ###

add_dependencies(Risk Shaders)
target_link_libraries(Risk glfw)
target_link_libraries(Risk Vulkan::Vulkan)
target_link_libraries(Risk glm)
target_link_libraries(Risk vkbootstrap)
target_link_libraries(Risk tinyobjloader)
target_link_libraries(Risk imgui)
target_link_libraries(Risk stb_image)


### COMPILE ARGUMENTS ###

target_compile_options(Risk PRIVATE "-Wall" "-Wextra") 

# target_compile_options(Vulkan-Engine PRIVATE "-Wall" "-Wextra" "-fsanitize=address" "-fno-omit-frame-pointer")
# target_link_options(Vulkan-Engine PRIVATE "-fsanitize=address")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  add_compile_options("-stdlib=libc++23")
endif()


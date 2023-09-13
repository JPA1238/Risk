#include "vk_mesh.h"

#include "tiny_obj_loader.h"
#include <iostream>

VertexInputDescription Vertex::get_vertex_description() {
    VertexInputDescription desc;

    // one vertex buffer binding with per-vertex rate
    VkVertexInputBindingDescription mainBinding = {};
    mainBinding.binding = 0;
    mainBinding.stride = sizeof(Vertex);
    mainBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    desc.bindings.push_back(mainBinding);

    // pos is stored at loc 0
    VkVertexInputAttributeDescription positionAttribute = {};
    positionAttribute.binding = 0;
    positionAttribute.location = 0;
    positionAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
    positionAttribute.offset = offsetof(Vertex, position);

    // normal at loc 1
    VkVertexInputAttributeDescription normalAttribute = {};
    normalAttribute.binding = 0;
    normalAttribute.location = 1;
    normalAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
    normalAttribute.offset = offsetof(Vertex, normal);

    // color at loc 2
    VkVertexInputAttributeDescription colorAttribute = {};
    colorAttribute.binding = 0;
    colorAttribute.location = 2;
    colorAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
    colorAttribute.offset = offsetof(Vertex, color);

    desc.attributes.push_back(positionAttribute);
    desc.attributes.push_back(normalAttribute);
    desc.attributes.push_back(colorAttribute);
    return desc;
}

bool Mesh::load_from_obj(std::string filename) {
    // contains the vertex arrays
    tinyobj::attrib_t attrib;
    // contains info for each separate obj
    std::vector<tinyobj::shape_t> shapes;
    // contains info avout the material for each shape
    // isn't used (yet)
    std::vector<tinyobj::material_t> materials;

    // warning and error output from load fnt
    std::string warn;
    std::string err;

    // load OBJ
    tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, const_cast<char*>(filename.c_str()), nullptr);
    
    if (!warn.empty()) {
        std::cout << "WARNING: " << warn << std::endl;
    }
    if (!err.empty()) {
        std::cerr << "ERROR: " << err << std::endl;
        return false;
    }

    // loop over shapes
    for (size_t s = 0; s < shapes.size(); s++) {
        size_t index_offset = 0;
        // loop over faces in shape
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            // hardcoded loading triangles
            const static uint fv = 3;

            // loop over vertices in face
            for (size_t v = 0; v < fv; v++) {
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

                tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
                tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
                tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];

                tinyobj::real_t nx = attrib.normals[3 * idx.normal_index + 0];
                tinyobj::real_t ny = attrib.normals[3 * idx.normal_index + 1];
                tinyobj::real_t nz = attrib.normals[3 * idx.normal_index + 2];

                Vertex vert;
                vert.position.x = vx;
                vert.position.y = vy;
                vert.position.z = vz;
                vert.normal.x = nx;
                vert.normal.y = ny;
                vert.normal.z = nz;

                vert.color = vert.normal;
                
                _vertices.push_back(vert);
            }
            index_offset += fv;
        }
    }

    return true;
}

// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

// std lib
#include <vector>
#include <deque>
#include <string>
#include <functional>
#include <unordered_map>

// own lib
#include "vk_types.h"
#include "vk_mesh.h"

// 3rd party lib
#include "glm/mat4x4.hpp"
#include "glm/vec4.hpp"

// variable
constexpr unsigned int FRAME_OVERLAP = 3;
const unsigned int MAX_OBJECTS = 50000;

// structs
struct DeletionQueue {
	std::deque<std::function<void()>> deletors;
	
	void push_function(std::function<void()>&& function) {
		deletors.push_back(function);
	}

	void flush() {
		// reverse iterate the deletion queueu to execute all the functions
		for (auto it = deletors.rbegin(); it != deletors.rend(); it++) {
			(*it)();
		}
		deletors.clear();
	}
};

struct MeshPushConstants {
	glm::vec4 data;
	glm::mat4 render_matrix;
};

struct Material {
	VkPipeline pipeline;
	VkPipelineLayout pipelineLayout;
};

struct RenderObject {
	Mesh *mesh;
	Material *material;
	glm::mat4 transformMatrix;
};

struct FrameData {
	VkSemaphore _presentSemaphore, _renderSemaphore;
	VkFence _renderFence;

	VkCommandPool _commandPool;
	VkCommandBuffer _mainCommandBuffer;

	AllocatedBuffer frameDataBuffer; // cam + scene buffer
	AllocatedBuffer objectBuffer;

	VkDescriptorSet globalDescriptor;
	VkDescriptorSet objectDescriptor;
};

struct GPUCameraData {
	glm::mat4 view;
	glm::mat4 proj;
	glm::mat4 viewproj;
};

struct GPUSceneData {
	glm::vec4 fogColor;	// w for exponent
	glm::vec4 fogDistance;	// x for min, y for max, zw unused
	glm::vec4 ambientColor;
	glm::vec4 sunlightDirection; // w for power
	glm::vec4 sunlightColor;
};

struct GPUObjectData {
	glm::mat4 modelMatrix;
};

struct SceneInfo {
	uint vertices;
	// TODO: Implement info calculations
	uint edges;
	uint faces;
	uint memorySize;
};

class VulkanEngine {
public:
	bool _isInitialized{ false };
	int _frameNumber{ 0 };

	glm::vec3 _camPos = {0.0f, 0.0f, 0.0f};

	VkExtent2D _windowExtent{ 1700 , 900 };
	struct GLFWwindow* _window{ nullptr };
	void init();
	void run();
	void draw();
	void cleanup();

	VkPhysicalDeviceProperties _gpuProperties;

	VkInstance _instance; // Vulkan library
	VkDebugUtilsMessengerEXT _debug_messenger; // Vulkan debug output handle
	VkPhysicalDevice _chosenGPU; // Chosen GPU
	VkDevice _device; // Vulkan device for commands
	VkSurfaceKHR _surface; // Vulkan window surface

	VkSwapchainKHR _swapchain;
	VkFormat _swapchainImageFormat; // image format expected for SDL
	std::vector<VkImage> _swapchainImages; // array of images from swapchain
	std::vector<VkImageView> _swapchainImageViews; // array of image-views from swapchain

	VkImageView _depthImageView;
	AllocatedImage _depthImage;
	VkFormat _depthFormat;

	VkQueue _graphicsQueue;
	uint32_t _graphicsQueueFamily;

	// frame storage
	FrameData _frames[FRAME_OVERLAP];
	FrameData& get_current_frame();

	VkRenderPass _renderPass;
	std::vector<VkFramebuffer> _framebuffers;

	GPUSceneData _sceneParameters;
	// AllocatedBuffer _sceneParameterBuffer;

	VkDescriptorPool _descriptorPool;
	
	VkDescriptorSetLayout _globalSetLayout;
	VkDescriptorSetLayout _objectSetLayout;

	VkPipelineLayout _meshPipelineLayout;
	VkPipeline _meshPipeline;
	Mesh _triangleMesh;
	Mesh _monkeyMesh;
	std::unordered_map<std::string, std::string> _meshFiles; // name | filename
	SceneInfo _sceneInfo;

	VmaAllocator _allocator; // vma lib allocator

	// default array of renderable objects
	std::vector<RenderObject> _renderables;
	
	std::unordered_map<std::string, Material> _materials;
	std::unordered_map<std::string, Mesh> _meshes;

	// functions
	Material* create_material(VkPipeline pipeline, VkPipelineLayout layout, const std::string& name);
	Material* get_material(const std::string& name);
	Mesh* get_mesh(const std::string& name);

private:
	const float CAMSPEED = 0.05f; // Does not depend on FPS
	int _selectedShader{ 0 };

	void init_vulkan();
	void init_swapchain();
	void init_commands();
	void init_default_renderpass();
	void init_framebuffers();
	void init_sync_structures();
	void init_descriptors();
	void init_pipelines();
	void init_scene();

	bool load_shader_module(const char* filePath, VkShaderModule* outShaderModule);

	void load_meshes();
	void upload_mesh(Mesh& mesh);

	void draw_objects(VkCommandBuffer cmd, RenderObject* first, int count);
	AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);
	size_t pad_uniform_buffer_size(size_t originalSize);

	DeletionQueue _mainDeletionQueue;
};

class PipelineBuilder {
public:
	// variables
	std::vector < VkPipelineShaderStageCreateInfo> _shaderStages;
	VkPipelineVertexInputStateCreateInfo _vertexInputInfo;
	VkPipelineInputAssemblyStateCreateInfo _inputAssembly;
	VkViewport _viewport;
	VkRect2D _scissor;
	VkPipelineRasterizationStateCreateInfo _rasterizer;
	VkPipelineColorBlendAttachmentState _colorBlendAttachment;
	VkPipelineMultisampleStateCreateInfo _multisampling;
	VkPipelineLayout _pipelineLayout;
	VkPipelineDepthStencilStateCreateInfo _depthStencil;

	// functions
	VkPipeline build_pipeline(VkDevice device, VkRenderPass pass);
};

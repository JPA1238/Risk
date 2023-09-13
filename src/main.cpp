#include "engine/vk_engine.h"

#include <GLFW/glfw3.h>
#include <numeric>
#include <iostream>

int main(int argc, char* argv[])
{
	VulkanEngine engine;

	engine.init();

	const uint AVGCOUNT = 60;
	double currentTime = glfwGetTime();
	double prevTime = currentTime;
	double FPS[AVGCOUNT];
	double FPSavg;

	while (!glfwWindowShouldClose(engine._window)) {
		glfwPollEvents();

		
		float CAMSPEED = 0.05f;

		if (glfwGetKey(engine._window, GLFW_KEY_A))
		{
			engine._camPos.x += CAMSPEED;
		}
		else if (glfwGetKey(engine._window, GLFW_KEY_D))
		{
			engine._camPos.x -= CAMSPEED;
		}
		if (glfwGetKey(engine._window, GLFW_KEY_LEFT_CONTROL))
		{
			engine._camPos.y += CAMSPEED;
		}
		else if (glfwGetKey(engine._window, GLFW_KEY_SPACE))
		{
			engine._camPos.y -= CAMSPEED;
		}
		if (glfwGetKey(engine._window, GLFW_KEY_W))
		{
			engine._camPos.z += CAMSPEED;
		}
		else if (glfwGetKey(engine._window, GLFW_KEY_S))
		{
			engine._camPos.z -= CAMSPEED;
		}

		if (std::fabs((glfwGetTime() - prevTime) - (1.0 / 60.0)) < 0.001) {
			engine.draw();

			FPS[engine._frameNumber % AVGCOUNT] = 1 / (currentTime - prevTime);

			if (engine._frameNumber % AVGCOUNT == 0) {
				FPSavg = std::accumulate(FPS, FPS + AVGCOUNT, FPSavg);
				std::cout << "FPS:" << FPSavg / AVGCOUNT << std::endl;
			}

			prevTime = currentTime;
		}

		currentTime = glfwGetTime();
	}

	engine.cleanup();	

	return 0;
}

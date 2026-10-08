# Build the VkFFT timing tool. Needs the headers from ../kspaceFirstOrder-Metal/fetch-deps-macos.sh.

THIRD_PARTY = ../kspaceFirstOrder-Metal/ThirdParty

CXX      = clang++
CXXFLAGS = -std=c++17 -O2 -w -isystem $(THIRD_PARTY)/metal-cpp -isystem $(THIRD_PARTY)/VkFFT/vkFFT
LDFLAGS  = -framework Metal -framework Foundation -framework QuartzCore

fftbench: fftbench.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f fftbench

# # 1. clone MNN
# git clone https://github.jobcher.com/gh/https://github.com/alibaba/MNN.git --depth=1

# 2. build MNN
cd MNN
mkdir build
cd build
cmake -DMNN_LOW_MEMORY=ON -DMNN_SUPPORT_BF16=ON -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_QUANTOOLS=ON -DMNN_ARM82=ON ..
make -j4
cd ../..

# 3. copy headers and libs
cp -r MNN/include/MNN include
# linux
cp MNN/build/libMNN.so MNN/build/express/libMNN_Express.so libs 2> /dev/null || :
# macos
cp MNN/build/libMNN.dylib MNN/build/express/libMNN_Express.dylib libs  2> /dev/null || :

# 4. build mnn-llm
mkdir build
cd build
cmake ..
make -j4
cd ..
for i in `seq 0 27`
do
    wget https://ghproxy.com/https://github.com/wangzhaode/ChatGLM-MNN/releases/download/v0.1/glm_block_$i.mnn
done

wget https://ghproxy.com/https://github.com/wangzhaode/ChatGLM-MNN/releases/download/v0.1/slim_lm.bin
wget https://ghproxy.com/https://github.com/wangzhaode/ChatGLM-MNN/releases/download/v0.1/slim_word_embeddings.bin
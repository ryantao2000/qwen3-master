package com.xllz;


import dev.langchain4j.model.ollama.OllamaChatModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
public class Qwen3Test {
    @Autowired
    private OllamaChatModel ollamaChatModel;
    @Test
    public void testOllama() {
        //思考模式
        String answer = ollamaChatModel.chat("你好,你是谁");

        // 禁止思考模式
//        String answer = ollamaChatModel.chat("你好,你是谁 /no_think");
        //输出结果
        System.out.println(answer);
    }

    @Autowired
    private OpenAiChatModel openAiChatModel;
    @Test
    public void testSpringBoot() {
        //向模型提问
        String answer = openAiChatModel.chat("你是谁");

        //输出结果
        System.out.println(answer);
    }

}

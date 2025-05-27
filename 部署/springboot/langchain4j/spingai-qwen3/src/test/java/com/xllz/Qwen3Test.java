package com.xllz;

import org.junit.jupiter.api.Test;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import reactor.core.publisher.Flux;

@SpringBootTest
public class Qwen3Test {


    @Autowired
    private  ChatClient chatClient;


    @Test
    public void testOllama() {

        Flux<String> answer = chatClient.prompt()
                .user("你是谁")
                .stream()
                .content();

// 订阅并打印每个数据块
        answer.subscribe(
                chunk -> System.out.print(chunk), // 处理每个数据块
                error -> System.err.println("Error: " + error), // 错误处理
                () -> System.out.println("\nStream completed") // 完成回调
        );
    }




}

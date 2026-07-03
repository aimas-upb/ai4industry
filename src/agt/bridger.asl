

/* Initial beliefs and rules */



/* Initial goals */

!say_hello.


/* Plans */

+!say_hello
   <-     
    makeArtifact("llmBridge", "LlmBridge", [], LLMB);
    focus(LLMB);
    solve
   .

+llmResult(Result)
  <-
    .print("llm result changed to: ", Result).
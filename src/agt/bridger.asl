

/* Initial beliefs and rules */



/* Initial goals */

!say_hello.


/* Plans */

+!say_hello
   <-     
    makeArtifact("llmBridge", "LlmBridge", [], LLMB);
    focus(LLMB);
    solve("!carry(\"APAS\", \"DX10_output\", \"XY10_input\")")
   .

+llmResult(Result)
  <-
    .print("llm result changed to: ", Result).
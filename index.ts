import { ChatOpenAI } from "@langchain/openai";
import {
  START,
  END,
  MessagesAnnotation,
  StateGraph,
  MemorySaver,
} from "@langchain/langgraph";
import { v4 as uuidv4 } from "uuid";
import { ChatPromptTemplate } from "@langchain/core/prompts";

// Create a unique thread ID
const config = { configurable: { thread_id: uuidv4() } };
const config2 = { configurable: { thread_id: uuidv4() } };

// Initialize the LLM
const llm = new ChatOpenAI({
  model: "gpt-4o-mini",
  temperature: 0,
});

// Create a prompt template
const promptTemplate = ChatPromptTemplate.fromMessages([
  [
    "system",
    "You talk like a pirate. Answer all questions to the best of your ability.",
  ],
  ["placeholder", "{messages}"],
]);

// Define the function that calls the model
const callModel = async (state: typeof MessagesAnnotation.State) => {
  const response = await llm.invoke(state.messages);
  return { messages: response };
};

const callModel2 = async (state: typeof MessagesAnnotation.State) => {
  const prompt = await promptTemplate.invoke(state);
  const response = await llm.invoke(prompt);
  return { messages: response };
};



// Define a new graph
const workflow = new StateGraph(MessagesAnnotation)
  // Define the node and edge
  .addNode("model", callModel)
  .addEdge(START, "model")
  .addEdge("model", END);

const workflow2 = new StateGraph(MessagesAnnotation)
  // Define the node and edge
  .addNode("model", callModel2)
  .addEdge(START, "model")
  .addEdge("model", END);

// Add memory
const memory = new MemorySaver();
const app = workflow.compile({ checkpointer: memory });
const app2 = workflow2.compile({ checkpointer: memory });

// Run the graph
const input = [
  {
    role: "user",
    content: "Hi! I'm Satya.",
  },
];

const input2 = [
  {
    role: "user",
    content: "What's my name?",
  },
];

const input3 = [
  {
    role: "user",
    content: "Hi! I'm Som.",
  },
];

const input4 = [
  {
    role: "user",
    content: "What's my name?",
  },
];

const output = await app.invoke({ messages: input }, config);
const output2 = await app.invoke({ messages: input2 }, config);
const output3 = await app2.invoke({ messages: input3 }, config2);
const output4 = await app2.invoke({ messages: input4 }, config2);
// The output contains all messages in the state.
// This will log the last message in the conversation.
console.log(output.messages[output.messages.length - 1]?.content);
console.log(output2.messages[output2.messages.length - 1]?.content);
console.log(output3.messages[output3.messages.length - 1]?.content);
console.log(output4.messages[output4.messages.length - 1]?.content);

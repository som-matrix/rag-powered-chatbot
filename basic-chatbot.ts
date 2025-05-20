import { ChatOpenAI } from "@langchain/openai";
import {
  START,
  END,
  MessagesAnnotation,
  StateGraph,
  MemorySaver,
  Annotation
} from "@langchain/langgraph";
import { v4 as uuidv4 } from "uuid";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import {
  SystemMessage,
  HumanMessage,
  AIMessage,
  trimMessages,
} from "@langchain/core/messages";

const trimmer = trimMessages({
  maxTokens: 10,
  strategy: "last",
  tokenCounter: (msgs) => msgs.length,
  includeSystem: true,
  allowPartial: false,
  startOn: "human",
});

// Create a unique thread ID
const config = { configurable: { thread_id: uuidv4() } };
const config2 = { configurable: { thread_id: uuidv4() } };
const config3 = { configurable: { thread_id: uuidv4() } };
const config4 = { configurable: { thread_id: uuidv4() } };

// Initialize the LLM
const llm = new ChatOpenAI({
  model: "gpt-4o-mini",
  temperature: 0,
});

// Define the State
const GraphAnnotation = Annotation.Root({
  ...MessagesAnnotation.spec,
  language: Annotation<string>(),
});

// Create a prompt template
const promptTemplate = ChatPromptTemplate.fromMessages([
  [
    "system",
    "You talk like a pirate. Answer all questions to the best of your ability.",
  ],
  ["placeholder", "{messages}"],
]);

const promptTemplate2 = ChatPromptTemplate.fromMessages([
  [
    "system",
    "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
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

const callModel3 = async (state: typeof GraphAnnotation.State) => {
  const prompt = await promptTemplate2.invoke({
    messages: state.messages,
    language: state.language,
  });
  const response = await llm.invoke(prompt);
  return { messages: response };
};

const callModel4 = async (state: typeof GraphAnnotation.State) => {
  const trimmedMessage = await trimmer.invoke(state.messages);
  const prompt = await promptTemplate2.invoke({
    messages: trimmedMessage,
    language: state.language,
  });
  const response = await llm.invoke(prompt);
  return { messages: [response] };
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

const workflow3 = new StateGraph(GraphAnnotation)
  // Define the node and edge
  .addNode("model", callModel3)
  .addEdge(START, "model")
  .addEdge("model", END);

const workflow4 = new StateGraph(GraphAnnotation)
  // Define the node and edge
  .addNode("model", callModel4)
  .addEdge(START, "model")
  .addEdge("model", END);

// Add memory
const memory = new MemorySaver();
const app = workflow.compile({ checkpointer: memory });
const app2 = workflow2.compile({ checkpointer: memory });
const app3 = workflow3.compile({ checkpointer: memory });
const app4 = workflow4.compile({ checkpointer: memory });

const messages = [
  new SystemMessage("you're a good assistant"),
  new HumanMessage("hi! I'm bob"),
  new AIMessage("hi!"),
  new HumanMessage("I like vanilla ice cream"),
  new AIMessage("nice"),
  new HumanMessage("whats 2 + 2"),
  new AIMessage("4"),
  new HumanMessage("thanks"),
  new AIMessage("no problem!"),
  new HumanMessage("having fun?"),
  new AIMessage("yes!"),
];

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

const input5 = {
  messages: [
    {
      role: "user",
      content: "Hi im satya",
    },
  ],
  language: "Spanish",
};

const input6 = {
  messages: [
    {
      role: "user",
      content: "What is my name?",
    },
  ],
};

const input7 = {
  messages: [...messages, new HumanMessage("what's up?")],
  language: "English",
};

const output = await app.invoke({ messages: input }, config);
const output2 = await app.invoke({ messages: input2 }, config);
const output3 = await app2.invoke({ messages: input3 }, config2);
const output4 = await app2.invoke({ messages: input4 }, config2);
const output5 = await app3.invoke(input5, config3);
const output6 = await app3.invoke(input6, config3);
const output7 = await app4.invoke(input7, config4);
// The output contains all messages in the state.
// This will log the last message in the conversation.
console.log(output.messages[output.messages.length - 1]?.content);
console.log(output2.messages[output2.messages.length - 1]?.content);
console.log(output3.messages[output3.messages.length - 1]?.content);
console.log(output4.messages[output4.messages.length - 1]?.content);
console.log(output5.messages[output5.messages.length - 1]?.content);
console.log(output6.messages[output6.messages.length - 1]?.content);
console.log(output7.messages[output7.messages.length - 1]?.content);

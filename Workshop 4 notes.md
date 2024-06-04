![[_attachments/Pasted image 20240604121002.png]]

Recap on the two types of fine tunes:

- Full fine tunes that change all weights
- LoRA


![[_attachments/Pasted image 20240604121018.png]]

Different ways to save the output, which will affect the way you deploy the model

![[_attachments/Pasted image 20240604121030.png]]

![[_attachments/Pasted image 20240604121109.png]]



![[_attachments/Pasted image 20240604121504.png]]

These models take a while to load into memory. Cold starts can take 30 seconds to.... eh, 8 minutes (whatever). These can add up to real costs if you're paying for GPUs by the minute/second

![[_attachments/Pasted image 20240604121640.png]]

Many projects aren't real time, even though ChatGPT and others give that impression today

![[_attachments/Pasted image 20240604121833.png]]

![[_attachments/Pasted image 20240604121903.png]]
Example workflow of merging LoRA to base after fine tuning

![[_attachments/Pasted image 20240604122029.png]]
After that, can push model files to HuggingFace

![[_attachments/Pasted image 20240604122153.png]]

Video demo showing how to deploy from HuggingFace to a hosted inference platform


Hamel takes over. Will walk through actual deployment of Honeycomb inference

![[_attachments/Pasted image 20240604122559.png]]

Different requirements to think about. More boxes checked on the righthand column = more complexity

![[_attachments/Pasted image 20240604122715.png]]

![[_attachments/Pasted image 20240604122740.png]]

Advanced serving has autoscaling, load balancing for high availability, etc

![[_attachments/Pasted image 20240604122929.png]]

![[_attachments/Pasted image 20240604123059.png]]

Don't pay too close of attention but at a high level this is useful.

You should try different inference servers to see what works for you. vLLM is easy to use. Nvidia stack is more performant, but hard to use.

![[_attachments/Pasted image 20240604123318.png]]

Going to walk through the Honeycomb case study

![[_attachments/Pasted image 20240604123338.png]]

Honeycomb needed to be realtime. Because users need to ask questions and get answers right away.

Hamel launched this on Replicate.

![[_attachments/Pasted image 20240604123411.png]]
This is what Replicate looks like.

Reason why, it's nice to have a playground for business users to be able to play around with.

Recall: two inputs into the language model:
1) The user's natural language query (NLQ)
2) The schema (which usually comes from RAG)

Hamel's counterpart Phillip was able to send him permalinks to data that looked "weird." Which let Hamel easily dig in.

Sometimes it's an easy data mistake like an extra newline.

Replicate also comes with bundled documentation. It even dynamically includes the playground data into code examples. Really easy.

![[_attachments/Pasted image 20240604123654.png]]

![[_attachments/Pasted image 20240604123807.png]]
Cog is a wrapper around Docker: https://github.com/replicate/cog

Helps avoid footguns with CUDA + Docker

![[_attachments/Pasted image 20240604123956.png]]

All you need are two pieces:
* cog.yaml
* predict.py

![[_attachments/Pasted image 20240604124057.png]]

Highlighted command downloads model weights really fast

![[_attachments/Pasted image 20240604124318.png]]

Have to create model on Replicate platform

Important thing is to pick the right-sized hardware/GPU

Select "Custom cog model"

![[_attachments/Pasted image 20240604124413.png]]

Back in you will have a fully-qualified name kind of like a GitHub repository

![[_attachments/Pasted image 20240604124516.png]]

Demonstration of how model was quantized using standard methods available on HuggingFace

Time to turn it over to Joe who can talk about Replicate.

![[_attachments/Pasted image 20240604124739.png]]

Have been workign with generative language models for four years. Things have changed a lot.

One thing that hasn't changed: it's really hard to deploy language models!

But maybe not in the ways you think.

![[_attachments/Pasted image 20240604124829.png]]

Performance is multidimensional and zero sum.

You need a simple model that outputs structured data, don't care too much about performance and cost: not too hard.

![[_attachments/Pasted image 20240604124949.png]]

You will be making tradeoffs from one dimension to another

![[_attachments/Pasted image 20240604125018.png]]

Used to be not too many options for serving. Just in the last two years or less, there are very many. And it keeps changing.

You don't need to always keep up, if you're serving models and it works fine. But little things like 10% improvements can come up that may be worth pursuing.

![[_attachments/Pasted image 20240604125118.png]]

The antidote to changing technology is minimize the cost of experimentation. Make it easy to change tech later.

How do we make it easy to serve language models?

![[_attachments/Pasted image 20240604125202.png]]

Lots of buzzwords that can be hard to wade through

![[_attachments/Pasted image 20240604125242.png]]

Just a couple things that make most things slow

![[_attachments/Pasted image 20240604125304.png]]

Transformers and neural networks transferring data from device memory to smaller device memory caches. Transferring data from slow memory to fast memory

![[_attachments/Pasted image 20240604125340.png]]

Workaround to this is to transfer less data

One fancy way to do this is CUDA kernels. Just functions that run on a GPU

Examples are softmax kernel, attention kernel, etc.

Flash attention is one that made a big splash

More efficient data storage, more efficient data transfer. It mostly comes down to better data management

Make data smaller - quantization, or speculative decoding.

Fundamentally these techniques are about More efficient data storage, more efficient data transfer. It mostly comes down to better data management


![[_attachments/Pasted image 20240604125553.png]]

Software overhead is the other pain point

Kernels are another antidote

![[_attachments/Pasted image 20240604125650.png]]

Recap - these are the two main bottlenecks of transformers

![[_attachments/Pasted image 20240604125711.png]]

Talked about the first column

KV caching = instead of having to re-encode tokens during chat prompting conversation, cache them

![[_attachments/Pasted image 20240604125845.png]]

This is an "older" technique ("ORCA" paper?)

Batching would be bad for real-time needs. Makes you wait until all requests fill up the GPU before response is generated

Continuous batching fixed this. Fixed two problems:
1) Don't need to wait for new requests to come in. Finished requests are pulled off as they complete. It does add some complexity in situations where you really care about performance. A consequence can be you end up with dynamic batch sizes which has huge implications on cost and performance of the mdoel
2) 

![[_attachments/Pasted image 20240604130156.png]]

Interfaces are a lot different, but many use the same optimization techniques (e.g. CUDA graphs)

![[_attachments/Pasted image 20240604130247.png]]

Things get tricky when you care about performance

Measuring performance gets tricky with LLMs and continuous batching

![[_attachments/Pasted image 20240604130332.png]]

Lots of platforms that make promises of performances "up to X". They're talking about maximum single-stream tokens per second, i.e. batch size of 1. Which nobody is running at, and you don't have control over. Batch size varies substantially.

Need to think about what you're doing. Do you want to optimize total tokens per second? Or single stream tokens per second. Agent-focused workflows would be total tokens per second

Anecdote: talked to a recent advanced company who provided total tokens per second but didn't know about single stream tokens per second

![[_attachments/Pasted image 20240604130952.png]]

Tradeoffs in performance tuning

![[_attachments/Pasted image 20240604131019.png]]

Thinking about making your stack modular. With a quick demo

![[_attachments/Pasted image 20240604131045.png]]

It's important to prioritize modularity. Many horror stories fall under this category

TRTLM is a framework that is pretty well built, has people working on features, fixing bugs, etc. But for a project, speculative decoding surprisingly didn't work with streaming. Worked, sort of, but not with his use case.

Time and again he has found features he needs that aren't implemented, or can't be enabled with certain mixtures of needs.

So it's really valuable to be able to change frameworks as needed. To that efficiently, you need to be able to experiment with different frameworks.

Bugs are there, not always well documented. Need to be agile

![[_attachments/Pasted image 20240604131317.png]]

This is one of the things they're trying to solve with Replicate. Simplifying this experimentation.

Gives you complete control over your serving framework.

He's using TRTLM in one. Another one in vLLM.

Direction going in Replicate is not only do you have many models, but also many serving frameworks.

You'll still have to get into the details if you care about performance tuning. But should be a lot easier to experiment with these frameworks to quickly see what's broken.

Close to open sourcing cog-trt-llm

![[_attachments/Pasted image 20240604131445.png]]

He wants the process of trying these things to be very easy.

Replicate is completely open source

![[_attachments/Pasted image 20240604131526.png]]

Going to demonstrate a workflow

![[_attachments/Pasted image 20240604131706.png]]

Video demo of Replicate

On Replicate a training is something that runs on a container that creates an artifact - some weights

Can pull from HuggingFace and get a model running on Replicate

![[_attachments/Pasted image 20240604132021.png]]
Trained model looks like this

![[_attachments/Pasted image 20240604132038.png]]

Support for different clients

![[_attachments/Pasted image 20240604132130.png]]

This is possible thanks to cog-vLLM. You can run this all locally - see README

![[_attachments/Pasted image 20240604132620.png]]

Example of serving the LLM locally with cog-vllm

Moving on to Travis

![[_attachments/Pasted image 20240604132657.png]]

Main theme of talk will be lessons learned building his platform on training and serving fine-tuned LLMs

![[_attachments/Pasted image 20240604132748.png]]

![[_attachments/Pasted image 20240604132840.png]]

Most sales-y slide about the Predibase platform. They try to be very end-to-end

![[_attachments/Pasted image 20240604132914.png]]

You're paying for the extra capacity in $, latency, whatever for overly-capable models.

![[_attachments/Pasted image 20240604133004.png]]

Common pattern they envision are migrations from GPT-4 to many fine-tuned models for different tasks

![[_attachments/Pasted image 20240604133029.png]]
![[_attachments/Pasted image 20240604133043.png]]

16 different use cases, could cost $14k a month to serve

If every user that comes in needs a dedicated platform to serve on, could be ver yexpensive 

![[_attachments/Pasted image 20240604133123.png]]

The old way of deployments

![[_attachments/Pasted image 20240604133150.png]]

Deploying same base model parameters over and over again. What if shared base models were abstracted away

![[_attachments/Pasted image 20240604133210.png]]

Built on top of HF TGI. Forked it a bit, used to support serverless inference on Predibase

![[_attachments/Pasted image 20240604133255.png]]

Efficiently batches requests across different LoRA adapters using CUDA adapters. See paper called Punica

![[_attachments/Pasted image 20240604133416.png]]

Compared to the baseline of naive adapter swapping, where throughput suffers, LoRAX's efficient method maintains throughput

![[_attachments/Pasted image 20240604133505.png]]

Throughput translates into cost savings

![[_attachments/Pasted image 20240604133559.png]]

Enough about Predibase. Let's talk about the newbie use case

![[_attachments/Pasted image 20240604133625.png]]

Lots of benefits to not merging

![[_attachments/Pasted image 20240604133919.png]]

![[_attachments/Pasted image 20240604133940.png]]

Problem with quantization used in training but not when inferencing

![[_attachments/Pasted image 20240604134023.png]]

Can fix this by using QLoRA for inference but it's **slow**

![[_attachments/Pasted image 20240604134117.png]]
![[_attachments/Pasted image 20240604134128.png]]

Better solution: dequantize using bitsandbytes

![[_attachments/Pasted image 20240604134201.png]]

Solution fixes quantization errors during inference

![[_attachments/Pasted image 20240604134246.png]]

![[_attachments/Pasted image 20240604134433.png]]

Hardware is an important reqmt to consider

Typical recommendation, back of envelope is 1.5x the model weights needed for serving the model. Because it's not only the model weights needed for serving. E.g. also need the activations

![[_attachments/Pasted image 20240604134525.png]]

![[_attachments/Pasted image 20240604134603.png]]

Useful checklist for considering requirements

![[_attachments/Pasted image 20240604134639.png]]

Serverless or dedicated? Consider the realities of latency and request volume of the two approaches vs your requirements

![[_attachments/Pasted image 20240604134811.png]]

Where he thinks things are headed

![[_attachments/Pasted image 20240604134831.png]]

Fine tuning is often measured for accuracy but not often for speed.

There is a performance hit for using a LoRA model

![[_attachments/Pasted image 20240604134920.png]]

Ways to make LoRA faster than the base model: speculative decoding

![[_attachments/Pasted image 20240604134944.png]]
Medusa paper. Reject tokens that are categorically incorrect.

![[_attachments/Pasted image 20240604135014.png]]

How to combine two approaches? Predibase is working on this. Called lookahead LoRA

![[_attachments/Pasted image 20240604135042.png]]

Same or better model perf

![[_attachments/Pasted image 20240604135052.png]]

But the big win is in throughput

![[_attachments/Pasted image 20240604135134.png]]

Quick demo of how this works in practice

![[_attachments/Pasted image 20240604135156.png]]

98 tok/sec generic Medusa model

![[_attachments/Pasted image 20240604135218.png]]

113 tok/sec

![[_attachments/Pasted image 20240604135234.png]]

147 tok/sec fine-tuned for task + throughput


![[_attachments/Pasted image 20240604135344.png]]

Now over to Charles

Talking about deploying LLMs to Modal
Told to cover batch vs. streaming but that seems to be covered already

![[_attachments/Pasted image 20240604135410.png]]

Fundamental tension of throughput vs. latency

![[_attachments/Pasted image 20240604135427.png]]

What is "slow?" Took a long time to do a big job? Or something very small

The former means not enough throughput. The latter means poor latency

When optimizing these two things, you can deploy resources to improve service levels

![[_attachments/Pasted image 20240604135528.png]]

Throughput = batch-oriented. E.g. nightly recommendation system refresh, e.g. spotify replay

Real-time is guardrails, chatbots, etc.

![[_attachments/Pasted image 20240604135630.png]]

Throughput - if you put a lot of pressure on an external logging system it might fail

Latency - almost all the time, low latency systems with LLMs is human perception.

Can use tricks, like showing drafts or lower quality result which gets replaced with higher quality result. Backed by psychophysical user research. As long as it's responding within a couple hundred milliseconds. Then human becomes the bottleneck

![[_attachments/Pasted image 20240604135827.png]]

Latency is what gets people excited. ChatGPT!

![[_attachments/Pasted image 20240604135859.png]]

Old phenomenon. Old network saying re: much easier to improve bandwidth than latency

Microprocessor got over 1,000x faster in bandwidth but only a bit over 10x faster in latency. You run into limitations of physics

![[_attachments/Pasted image 20240604140029.png]]

GPUs are throughput-oriented. CPUs are latency-oriented.

This is reflected in physical silicon areas on the devices

![[_attachments/Pasted image 20240604140502.png]]

Easy to increase LLM throughput. Near impossible to increase latency.

Groq replaces the memory in a GPU with CPU-cache level hardware, which improves latency but increases cost


![[_attachments/Pasted image 20240604140539.png]]

![[_attachments/Pasted image 20240604140605.png]]

davinci-002 in 2022 to llama-3-8b in 2024 = 100x decrease in cost over two years for same "cognition" level

Bad idea to run an inference as a service platform right now - it's a race to the bottom

![[_attachments/Pasted image 20240604140913.png]]

What's the current story of deploying on Modal?

![[_attachments/Pasted image 20240604141233.png]]


![[_attachments/Pasted image 20240604141317.png]]

Modal is more than just for GPU inference: it's a whole serverless runtime for running lots of different workloads

![[_attachments/Pasted image 20240604141429.png]]

Demo of a program running on Modal that grants credits to people

![[_attachments/Pasted image 20240604141543.png]]

![[_attachments/Pasted image 20240604141917.png]]

It worked!

![[_attachments/Pasted image 20240604142034.png]]

Example of running obliterated Llama on Modal, an orthogonalized LLM. Orthogonalizing removes ability to say no from open weights

![[_attachments/Pasted image 20240604142146.png]]

![[_attachments/Pasted image 20240604142214.png]]

After a little while, the model is still cold starting. Latency is a common problem with these platforms

![[_attachments/Pasted image 20240604142244.png]]
In the meantime can check the logs to see what's going on. He said all these platforms run off the same open source tools

![[_attachments/Pasted image 20240604142423.png]]

Another example of serving trtllm model on Modal

![[_attachments/Pasted image 20240604142654.png]]

They have OpenAI-compatible endpoints. Can be used by tools like Instructor that expects that format




## Q&A questions

![[_attachments/Pasted image 20240604142909.png]]

![[_attachments/Pasted image 20240604142925.png]]

![[_attachments/Pasted image 20240604142946.png]]

![[_attachments/Pasted image 20240604143008.png]]

![[_attachments/Pasted image 20240604143034.png]]

![[_attachments/Pasted image 20240604143102.png]]

![[_attachments/Pasted image 20240604143129.png]]

![[_attachments/Pasted image 20240604143150.png]]


import React from 'react';
import type { Cell } from './types';

// SVG Icons (re-used for new components)
export const PlayIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
  </svg>
);

export const TerminalIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M8 9l4-4 4 4m0 6l-4 4-4-4" />
    </svg>
);


export const NOTEBOOK_DATA: Cell[] = [
    {
        type: 'markdown',
        content: {
            level: 'h1',
            title: "Introduction",
            description: "The POST (Privacy Of Soft-prompt Transfer) framework enables users to securely adapt large language models (LLMs) with private data. This interactive notebook breaks down the implementation into three core stages, with runnable Python code cells to simulate the key operations."
        }
    },
    {
        type: 'markdown',
        content: {
            level: 'h2',
            title: "Stage 1: Knowledge Distillation (KD)",
            description: "First, we create a smaller 'student' model from a large 'teacher' LLM. The student learns to mimic the teacher's behavior on a public dataset, making it efficient for local use. The following cells simulate loading models and data using the Hugging Face libraries `transformers` and `datasets`."
        }
    },
    {
        type: 'code',
        content: `print("--- Simulating Model Loading with 'transformers' ---")

# In a real environment, you'd run:
# from transformers import AutoModelForCausalLM
# teacher_model = AutoModelForCausalLM.from_pretrained("gpt2-large")
# student_model = AutoModelForCausalLM.from_pretrained("gpt2")

# For this browser-based simulation, we'll create mock classes
# to represent the models without downloading gigabytes of data.
class MockHuggingFaceModel:
    def __init__(self, name, num_layers):
        self.name = name
        self.num_layers = num_layers
        print(f"-> Mock model '{self.name}' initialized with {self.num_layers} layers.")

teacher_model = MockHuggingFaceModel("gpt2-large", 36)
student_model = MockHuggingFaceModel("gpt2", 12)

print("\\nSimulation complete. Models are ready in the environment.")`
    },
    {
        type: 'code',
        content: `print("--- Simulating Data Loading with 'datasets' ---")

# In a real environment, you'd run:
# from datasets import load_dataset
# distillation_dataset = load_dataset("bookcorpus", split="train")

# For this simulation, we create a small, representative list of strings.
class MockDataset:
    def __init__(self, data):
        self._data = data
        print(f"-> Mock dataset created with {len(data)} items.")

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self):
        return len(self._data)

mock_data = [
    {"text": "This is a sentence from our public corpus."},
    {"text": "Here is another sentence for the distillation process."},
]
distillation_dataset = MockDataset(mock_data)

print(f"First item: {distillation_dataset[0]}")
print("\\nSimulation complete. Dataset is ready.")`
    },
    {
        type: 'markdown',
        content: {
            level: 'h3',
            title: "Method: Weight Initialization",
            description: "To speed up convergence, we initialize the student model's layers with corresponding weights taken from the teacher model. This code cell simulates this process using PyTorch modules."
        }
    },
    {
        type: 'code',
        content: `import torch
import torch.nn as nn
from collections import OrderedDict

print("--- Simulating Weight Initialization ---")

# 1. Define simplified Teacher and Student model architectures
class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Sequential(OrderedDict([
            ('layer_0', nn.Linear(10, 10)),
            ('layer_1', nn.Linear(10, 10)),
            ('layer_2', nn.Linear(10, 10)),
        ]))
        self.lm_head = nn.Linear(10, 5)

class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Sequential(OrderedDict([
            ('layer_0', nn.Linear(10, 10)),
        ]))
        self.lm_head = nn.Linear(10, 5)

teacher = Teacher()
student = Student()

# 2. Check a weight before initialization
print("\\nStudent layer_0 weight (before):\\n", student.transformer.layer_0.weight.data[0, :5])

# 3. Perform initialization
teacher_state_dict = teacher.state_dict()
student_state_dict = student.state_dict()

copied_layers = 0
for name, param in teacher_state_dict.items():
    if name in student_state_dict and param.shape == student_state_dict[name].shape:
        student_state_dict[name].copy_(param)
        copied_layers += 1

student.load_state_dict(student_state_dict)
print(f"\\n-> Copied weights for {copied_layers} matching layers.")

# 4. Verify the weight after initialization
print("\\nStudent layer_0 weight (after):\\n", student.transformer.layer_0.weight.data[0, :5])
print("Teacher layer_0 weight:\\n", teacher.state_dict()['transformer.layer_0.weight'][0, :5])
print("\\nVerification complete. Weights now match.")`
    },
    {
        type: 'markdown',
        content: {
            level: 'h2',
            title: "Stage 2: Private Prompt Tuning",
            description: "The user tunes a soft prompt on the small student model locally using their private data. We use `peft` to add the prompt and `opacus` to add differential privacy guarantees."
        }
    },
    {
        type: 'code',
        content: `print("--- Simulating Soft Prompt Setup with 'peft' ---")

# In a real environment, you would use the actual peft library
# on a loaded transformers model.
# from peft import get_peft_model, PromptTuningConfig, TaskType

# Mock classes for simulation
class MockStudentModel:
    def __init__(self):
        self.name = "gpt2"
    def add_adapter(self, config):
        print(f"-> Soft prompt adapter added with {config['num_virtual_tokens']} virtual tokens.")
    def print_trainable_parameters(self):
         print("trainable params: 51,200 || all params: 124,490,240 || trainable%: 0.0411")


mock_student = MockStudentModel()

prompt_config = {
    "task_type": "CAUSAL_LM",
    "num_virtual_tokens": 50
}

# Simulate wrapping the model
mock_student.add_adapter(prompt_config)
mock_student.print_trainable_parameters()
print("\\nSimulation complete. Model is configured for prompt tuning.")`
    },
    {
        type: 'code',
        content: `import torch

print("--- Simulating Differentially Private Training with 'opacus' ---")

# In a real environment, you would use the actual opacus library.
# from opacus import PrivacyEngine

# Mock classes for simulation
class MockPeftModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10,1)
    def forward(self, x):
        return self.layer(x)

class MockPrivacyEngine:
    def make_private(self, module, optimizer, data_loader, noise_multiplier, max_grad_norm):
        print("-> Opacus PrivacyEngine attached.")
        print(f"   - Noise Multiplier: {noise_multiplier}")
        print(f"   - Max Grad Norm: {max_grad_norm}")
        return module, optimizer, data_loader
    def get_epsilon(self, delta):
        return 1.25 # Return a simulated privacy budget

# Setup
mock_model = MockPeftModel()
mock_optimizer = torch.optim.Adam(mock_model.parameters())
mock_loader = [(torch.randn(4,10), torch.randn(4,1))]
privacy_engine = MockPrivacyEngine()

# Simulate attaching the privacy engine
model, optimizer, data_loader = privacy_engine.make_private(
    module=mock_model,
    optimizer=mock_optimizer,
    data_loader=mock_loader,
    noise_multiplier=1.1,
    max_grad_norm=1.0,
)

epsilon = privacy_engine.get_epsilon(delta=1e-5)
print(f"\\nAchieved privacy guarantee (simulated): ε = {epsilon:.2f}")
print("The optimizer is now set up to add noise and clip gradients during training.")`
    },
     {
        type: 'markdown',
        content: {
            level: 'h2',
            title: "Stage 3: Privacy-Preserving Prompt Transfer",
            description: "The LLM provider transfers the user's tuned prompt to the large teacher model using only public data. This requires a custom loss function to align the *behavior* of the prompt on both models."
        }
    },
    {
        type: 'code',
        content: `import torch
import torch.nn.functional as F

print("--- Simulating the Custom Transfer Loss Function ---")

# We need tensors to represent model outputs (logits)
# Shape: [batch_size, sequence_length, vocab_size]
batch, seq_len, vocab = 2, 5, 10

# 1. Define mock model outputs
# The prompted large model's output is slightly different from the small one
prompted_large_logits = torch.randn(batch, seq_len, vocab)
prompted_small_logits = prompted_large_logits + torch.randn(batch, seq_len, vocab) * 0.1

# The 'impact' is the change caused by the prompt
unprompted_large_logits = prompted_large_logits - torch.randn(batch, seq_len, vocab) * 0.5
unprompted_small_logits = prompted_small_logits - torch.randn(batch, seq_len, vocab) * 0.4

# 2. Implement the transfer loss
def transfer_loss(pl_logits, ps_logits, ul_logits, us_logits, w=0.5):
    # Behavior Mimicking Loss: Make prompted models behave similarly
    mimic_loss = F.kl_div(
        F.log_softmax(pl_logits, dim=-1),
        F.softmax(ps_logits, dim=-1),
        reduction='batchmean'
    )
    
    # Impact Alignment Loss: Make the *effect* of the prompt similar
    large_impact = pl_logits - ul_logits
    small_impact = ps_logits - us_logits
    align_loss = F.kl_div(
        F.log_softmax(large_impact, dim=-1),
        F.softmax(small_impact, dim=-1),
        reduction='batchmean'
    )
    
    return w * mimic_loss + (1 - w) * align_loss

# 3. Calculate the loss
loss = transfer_loss(
    prompted_large_logits, prompted_small_logits,
    unprompted_large_logits, unprompted_small_logits
)

print(f"Behavior Mimicking Loss (simulated): {loss.item():.4f}")
print("\\nThis loss would be used to train *only* the new prompt on the large model.")`
    }
];

import dspy

class RAGModule(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate = dspy.ChainOfThought('context, question -> answer')

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)

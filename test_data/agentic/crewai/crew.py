from crewai import Agent, Crew, Task
from crewai_tools import FirecrawlScrapeWebsiteTool

class ResearchCrew:
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def crew(self):
        return Crew(agents=self.agents, tasks=self.tasks, verbose=True)

import sys
import os
import json
import math
import hashlib
import time
import numpy as np
import urllib.request
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# =============================================================================
# 1. CONFIG
# =============================================================================
def load_config(path=None):
    if path is None:
        import sys
        path = sys.argv[1] if len(sys.argv) > 1 else "gpt_mini3.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def config_hash(cfg):
    hashable = {k: v for k, v in cfg.items() if k in ("model", "tokenizer")}
    canonical = json.dumps(hashable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

# =============================================================================
# 2. DATA: Wikipedia download + word-level tokenization
# =============================================================================
class WordTokenizer:
    def __init__(self, max_vocab_size: int = 20000, max_word_len: int = 20):
        self.max_vocab_size = max_vocab_size
        self.max_word_len = max_word_len
        self.word2idx = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
        self.idx2word = {0: "<pad>", 1: "<unk>", 2: "<eos>"}
        self.vocab_size = 3

    def save(self, path):
        import json
        Path(path).write_text(json.dumps({"word2idx": self.word2idx, "idx2word": self.idx2word,
                                            "vocab_size": self.vocab_size,
                                            "max_vocab_size": self.max_vocab_size,
                                            "max_word_len": self.max_word_len}))

    def load(self, path):
        import json
        data = json.loads(Path(path).read_text())
        self.word2idx = data["word2idx"]
        self.idx2word = {int(k): v for k, v in data["idx2word"].items()}
        self.vocab_size = data["vocab_size"]
        self.max_vocab_size = data.get("max_vocab_size", self.max_vocab_size)
        self.max_word_len = data.get("max_word_len", self.max_word_len)

    def build_vocab(self, texts: list[str]):
        freq: dict[str, int] = {}
        total = len(texts)
        for i, text in enumerate(texts):
            if (i + 1) % 2000000 == 0 or i == total - 1:
                print(f"  Building vocab: {i+1}/{total} texts ({(i+1)*100//total}%)", flush=True)
            for word in self._tokenize_text(text):
                if len(word) > self.max_word_len:
                    continue
                freq[word] = freq.get(word, 0) + 1

        print(f"  Sorting {len(freq)} unique words...")
        sorted_words = sorted(freq.items(), key=lambda x: -x[1])
        for word, _ in sorted_words:
            if len(self.word2idx) >= self.max_vocab_size:
                break
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        self.vocab_size = len(self.word2idx)
        print(f"Vocabulary size: {self.vocab_size} (max: {self.max_vocab_size})")

    def _tokenize_text(self, text: str) -> list[str]:
        return text.lower().split()

    def encode(self, text: str) -> list[int]:
        unk = self.word2idx["<unk>"]
        return [self.word2idx.get(w, unk) for w in self._tokenize_text(text) if len(w) <= self.max_word_len]

    def decode(self, indices: list[int]) -> str:
        return " ".join(self.idx2word.get(i, "<unk>") for i in indices)


class WordDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer: WordTokenizer, seq_length: int, cache_file=None, device=None):
        import re, pickle
        print(f"  Creating dataset from {len(texts)} texts...")
        eos = tokenizer.word2idx["<eos>"]
        unk = tokenizer.word2idx["<unk>"]
        maxlen = tokenizer.max_word_len
        total = len(texts)
        strip_chars = ".,!?;:\"'()[]{},!?:"

        # Cache: save/load tokenized array to avoid reprocessing
        if cache_file is None:
            cache_file = Path("data_cache.npy")
        else:
            cache_file = Path(cache_file)
        if cache_file.exists() and (cache_file.stat().st_size > 1_000_000_000):
            print(f"  Loading cached dataset ({cache_file.stat().st_size // 1_000_000_000}GB)...", flush=True)
            arr = np.load(str(cache_file))
            print(f"  Cache hit: {len(arr)} tokens loaded", flush=True)
        else:
            # Estimate size and pre-allocate
            est = 0
            for text in texts:
                est += len(text.split()) + 1
            est = int(est * 1.1)
            arr = np.empty(est, dtype=np.int32)
            pos = 0

            for i, text in enumerate(texts):
                for w in text.lower().split():
                    w = w.strip(strip_chars)
                    if 1 <= len(w) <= maxlen:
                        arr[pos] = tokenizer.word2idx.get(w, unk)
                        pos += 1
                arr[pos] = eos
                pos += 1
                if (i + 1) % 2000000 == 0 or i == total - 1:
                    print(f"    {i+1}/{total} texts, {pos//1_000_000}M tokens", flush=True)

            arr = arr[:pos]
            print(f"  Saving cache ({pos//1_000_000}M tokens)...", flush=True)
            np.save(str(cache_file), arr)

        print(f"  Keeping {len(arr)//1_000_000}M tokens on CPU (transfers per batch)...", flush=True)
        self.data = arr  # keep as numpy array on CPU
        del arr
        self.seq_length = seq_length
        print(f"  Dataset ready: {len(self)} samples", flush=True)

    def __len__(self):
        return max(0, len(self.data) - self.seq_length)

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1:idx + self.seq_length + 1]
        return x, y


BUILTIN_CORPUS = """
In the beginning was the word, and the word was with God, and the word was God.
He was in the beginning with God. All things were made by him; and without him
was not any thing made that was made. In him was life; and the life was the
light of men. And the light shineth in darkness; and the darkness comprehended
it not. There was a man sent from God, whose name was John. The same came for
a witness, to bear witness of the Light, that all men through him might believe.
The Bible is a sacred book. It tells us about God and Jesus.
The sky is blue. The grass is green. The sun is hot.
I love learning about artificial intelligence and deep learning.
PyTorch is a powerful library for machine learning.
Transformers are the foundation of modern language models.
Attention mechanisms allow models to focus on relevant parts of the input.
Self-attention computes relationships between all positions in a sequence.
Deep learning has revolutionized natural language processing.
Neural networks can learn complex patterns from data.
Training large language models requires significant computational resources.
The future of AI is bright and full of possibilities.
Language models generate text that is coherent and meaningful.
They can write stories, poems, code, and essays.
Artificial intelligence is changing the world.
It is important to use AI responsibly and ethically.
We must ensure that AI benefits all of humanity.
Technology advances rapidly in the modern era.
Innovation drives progress in society.
Education is key to understanding new technologies.
Curiosity fuels the desire to learn and explore.
Knowledge is power in the digital age.
Data is the new oil in the information economy.
Algorithms process data to make predictions and decisions.
Machine learning models improve over time with more data.
Supervised learning uses labeled data for training.
Unsupervised learning finds patterns in unlabeled data.
Reinforcement learning learns through trial and error.
Deep reinforcement learning combines neural networks with RL.
AlphaGo defeated the world champion in Go.
This was a milestone in AI history.
Robots can now perform complex tasks in factories.
Self-driving cars are becoming a reality.
Medical AI helps diagnose diseases early.
AI assistants help us manage our daily lives.
Chatbots provide customer support around the clock.
Virtual reality creates immersive experiences.
Augmented reality overlays digital information on the real world.
The metaverse is a vision of a shared virtual space.
Blockchain technology ensures secure and transparent transactions.
Cryptocurrencies are digital assets secured by cryptography.
Bitcoin was the first cryptocurrency.
Ethereum enables smart contracts on the blockchain.
Decentralized finance aims to disrupt traditional banking.
Non-fungible tokens represent unique digital items.
Artificial general intelligence remains a long-term goal.
Narrow AI excels at specific tasks.
AI safety is a critical field of research.
We must align AI systems with human values.
Collaboration between humans and AI can lead to great outcomes.
Creativity and empathy are uniquely human traits.
AI can augment human capabilities, not replace them.
The journey of discovery is ongoing and exciting.
Every day brings new breakthroughs in science and technology.
We stand on the shoulders of giants who came before us.
Let us build a better future together with AI.
The universe is vast and full of wonders.
Stars are born from clouds of gas and dust.
Gravity holds planets in orbit around stars.
The human brain is the most complex organ.
Science seeks to understand the natural world.
Mathematics is the language of the universe.
History teaches us about the past.
Philosophy asks the big questions about existence.
Music is a universal form of expression.
Art reflects the human condition.
Literature allows us to explore different perspectives.
Science fiction imagines possible futures.
Poetry captures emotion in condensed form.
Dance is movement set to rhythm.
The ocean covers most of the Earth.
Mountains rise from the forces of the Earth.
Forests are home to countless species.
Rivers flow from mountains to the sea.
Deserts are some of the harshest environments.
Weather patterns shape our daily lives.
Climate change is one of the greatest challenges.
Renewable energy offers a sustainable future.
Solar power harnesses energy from the sun.
Wind turbines generate electricity from air currents.
Nuclear power produces massive amounts of energy.
Fossil fuels are the main cause of pollution.
Recycling helps reduce waste and conserve resources.
Conservation protects endangered species and habitats.
The rainforest is the lungs of the planet.
Coral reefs support marine biodiversity.
Polar ice caps are melting at an alarming rate.
Ocean acidification threatens marine ecosystems.
Biodiversity is essential for ecosystem health.
Humans are part of nature, not separate from it.
We must protect the planet for future generations.
Sustainable development balances economy and environment.
Green technology is transforming industries.
Electric vehicles reduce carbon emissions.
Smart cities use technology to improve quality of life.
Urban planning shapes how we live together.
Public transportation reduces traffic congestion.
Architecture shapes the character of a city.
Design influences how we interact with objects.
Engineering solves problems and builds infrastructure.
Programming creates the digital world.
Software powers everything from phones to rockets.
The internet connects billions of people.
Social media changes how we communicate.
E-commerce has transformed shopping.
Cloud computing stores data remotely.
Cybersecurity protects against digital threats.
Artificial intelligence learns from data.
Natural language processing enables machines to understand text.
Computer vision allows machines to see images.
Robotics combines mechanical and intelligent systems.
Automation increases efficiency and productivity.
The workforce is evolving with technology.
Remote work changes the nature of employment.
Education adapts to new technologies.
Online learning makes knowledge accessible.
Critical thinking is an essential skill.
Creativity is valued in the age of AI.
Empathy and emotional intelligence matter more than ever.
Leadership inspires people to achieve great things.
Teamwork combines diverse strengths.
Innovation requires taking risks.
Failure is a stepping stone to success.
Perseverance overcomes obstacles.
Discipline builds character.
Patience yields results.
Humility keeps us grounded.
Gratitude enriches our lives.
Kindness costs nothing but means everything.
Compassion heals wounds and builds bridges.
Forgiveness frees the soul.
Love is the most powerful force.
Hope sustains us through hard times.
Faith gives strength in adversity.
Courage faces fear head on.
Wisdom comes from experience and reflection.
Truth is the foundation of trust.
Justice ensures fairness and equality.
Freedom requires responsibility.
Democracy gives voice to the people.
Law upholds order and protects rights.
Ethics guide our decisions and actions.
Integrity means doing the right thing.
Honesty builds strong relationships.
Trust is earned over time.
Respect treats everyone with dignity.
Generosity shares abundance with others.
Service contributes to the common good.
Peace is the ultimate goal of humanity.
War brings suffering and destruction.
Conflict resolution requires dialogue and understanding.
Cooperation achieves what competition cannot.
Unity in diversity makes us stronger.
Cultural exchange enriches all societies.
Language connects people across borders.
Tradition preserves wisdom from the past.
Progress builds on the foundation of history.
Change is the only constant in life.
Adaptation is key to survival.
Evolution shapes the diversity of life.
Genetics determines our inherited traits.
Environment influences our development.
Health is the greatest wealth.
Exercise strengthens body and mind.
Nutrition fuels our daily activities.
Sleep restores energy and repairs cells.
Mental health is as important as physical health.
Stress management improves quality of life.
Mindfulness brings awareness to the present moment.
Meditation calms the restless mind.
Yoga combines movement with breath.
Music therapy aids emotional healing.
Art therapy expresses inner feelings.
Writing helps process complex emotions.
Journaling provides clarity and self-reflection.
Reading expands the mind and imagination.
Travel broadens perspectives and horizons.
Adventure fuels the spirit of exploration.
Discovery opens new possibilities.
Curiosity drives scientific inquiry.
Experimentation tests hypotheses and theories.
Observation reveals patterns in nature.
Analysis breaks down complex problems.
Synthesis combines ideas into new insights.
Communication shares knowledge with others.
Teaching multiplies understanding.
Mentorship guides the next generation.
Apprenticeship builds practical skills.
Research pushes the boundaries of knowledge.
Invention creates tools and technologies.
Discovery reveals hidden truths.
Innovation transforms ideas into reality.
Collaboration accelerates progress.
Competition drives excellence.
Creativity brings fresh perspectives.
Imagination envisions what could be.
Vision charts the course for the future.
Strategy plans the path to success.
Execution turns plans into results.
Leadership inspires action and change.
Management organizes resources and people.
Operations keep systems running smoothly.
Finance manages money and investments.
Marketing connects products with customers.
Sales generates revenue and growth.
Customer service builds loyalty and trust.
Quality assurance maintains high standards.
Continuous improvement drives excellence.
Lean thinking eliminates waste and inefficiency.
Agile methods adapt to changing requirements.
Scrum organizes work in short cycles.
Kanban visualizes workflow and bottlenecks.
DevOps bridges development and operations.
Cloud infrastructure scales with demand.
Microservices break monoliths into components.
APIs enable system communication.
Databases store and retrieve information.
Algorithms process data efficiently.
Data structures organize information logically.
Object-oriented programming models real-world entities.
Functional programming emphasizes pure functions.
Recursion solves problems by breaking them down.
Iteration repeats steps until a condition is met.
Abstraction hides complexity behind simple interfaces.
Encapsulation bundles data and behavior together.
Inheritance reuses code through class hierarchies.
Polymorphism allows one interface for many types.
Testing verifies that code works correctly.
Debugging finds and fixes errors in code.
Documentation explains how systems work.
Version control tracks changes over time.
Code review catches bugs and shares knowledge.
Refactoring improves code without changing behavior.
Optimization makes code run faster and use less memory.
Scalability handles growing demand gracefully.
Reliability ensures systems work under pressure.
Security protects against threats and vulnerabilities.
Privacy safeguards personal information.
Accessibility makes technology available to everyone.
Sustainability reduces environmental impact.
Ethics guides responsible technology use.
Governance establishes rules and accountability.
Compliance meets legal and regulatory requirements.
Transparency builds public trust.
Accountability ensures responsibility for outcomes.
Inclusivity welcomes diverse voices and perspectives.
Equity gives everyone fair opportunities.
Justice corrects systemic inequalities.
Rights protect individuals from oppression.
Liberty allows freedom of choice and expression.
Democracy distributes power among the people.
Republic balances freedom with structured governance.
Constitution limits government and protects rights.
Bill of rights enumerates fundamental freedoms.
Separation of powers prevents concentration of authority.
Checks and balances keep each branch accountable.
Rule of law applies equally to all citizens.
Due process protects against arbitrary government action.
Habeas corpus prevents unlawful detention.
Freedom of speech allows open discourse.
Freedom of religion protects spiritual beliefs.
Freedom of assembly enables peaceful protest.
Right to privacy shields personal information.
Right to education ensures equal opportunity.
Right to health care maintains public well-being.
Right to a fair trial protects the accused.
Right to vote gives citizens political voice.
Social contract binds citizens and government together.
Civic duty requires participation in democracy.
Voting is the foundation of representative government.
Public service contributes to society.
Volunteering helps communities thrive.
Charity alleviates suffering and poverty.
Philanthropy funds research and development.
Investment builds infrastructure and creates jobs.
Entrepreneurship drives economic innovation.
Small businesses form the backbone of economy.
Trade exchanges goods and services globally.
Supply chains connect producers and consumers.
Logistics manages the flow of goods.
Manufacturing transforms raw materials into products.
Agriculture feeds the world population.
Fisheries harvest resources from the ocean.
Forestry manages timber and wood products.
Mining extracts minerals and metals from Earth.
Energy production powers modern civilization.
Transportation moves people and goods.
Aviation connects distant parts of the world.
Shipping carries cargo across oceans.
Railways provide efficient land transport.
Roads form the backbone of surface travel.
Bridges span rivers and valleys.
Tunnels cut through mountains and underground.
Highways enable long-distance travel.
Airports facilitate international travel.
Harbors serve maritime commerce.
Dams control water flow and generate power.
Aqueducts transport water across distances.
Pipelines carry oil and gas overland.
Cables transmit data and power underwater.
Satellites orbit Earth for communication.
Telescopes observe distant stars and galaxies.
Microscopes reveal the microscopic world.
Particle accelerators explore subatomic physics.
Computers process information at incredible speeds.
Supercomputers simulate complex phenomena.
Quantum computers promise exponential speedup.
Neural networks learn from vast datasets.
Deep learning powers image and speech recognition.
Reinforcement learning trains agents through rewards.
Generative models create new content.
Large language models understand and generate text.
Knowledge graphs organize structured information.
Recommendation systems suggest relevant content.
Search engines find information in milliseconds.
Navigation systems guide drivers and pilots.
Drones perform aerial surveillance and delivery.
Robots automate repetitive and dangerous tasks.
Exoskeletons augment human strength.
Prosthetics restore lost physical functions.
Wearables track health and fitness metrics.
Smartphones connect us to the digital world.
Tablets provide portable computing.
Laptops enable mobile productivity.
Desktops handle intensive computing tasks.
Servers host websites and applications.
Data centers store massive amounts of information.
Cloud platforms deliver computing on demand.
Edge computing processes data closer to users.
IoT connects billions of devices worldwide.
Smart homes automate daily living.
Smart cities optimize urban infrastructure.
Autonomous vehicles navigate without human input.
Drones deliver packages to remote locations.
3D printing creates objects layer by layer.
Nanotechnology manipulates matter at atomic scale.
Biotechnology modifies living organisms.
Genetic engineering edits DNA sequences.
Stem cell research holds promise for regenerative medicine.
Vaccines prevent infectious diseases.
Antibiotics treat bacterial infections.
Immunotherapy harnesses the immune system to fight cancer.
Radiation therapy targets cancerous cells.
Chemotherapy kills rapidly dividing cells.
Surgery removes diseased tissue.
Transplantation replaces failing organs.
Diagnostics identify diseases early.
Imaging visualizes internal body structures.
MRI uses magnetic fields to create detailed images.
CT scans combine X-rays for cross-sectional views.
Ultrasound uses sound waves for real-time imaging.
X-rays reveal bone fractures and abnormalities.
Blood tests measure chemical markers.
Biopsy examines tissue samples.
Genetic testing identifies inherited conditions.
Screening detects diseases before symptoms appear.
Prevention reduces risk of illness.
Screening saves lives through early detection.
Treatment alleviates symptoms and cures disease.
Rehabilitation restores function after injury.
Palliative care comforts those with serious illness.
End-of-life care honors patient wishes.
Grief counseling supports those who mourn.
Support groups connect people with shared experiences.
Therapy addresses mental health challenges.
Counseling provides guidance during difficult times.
Mediation resolves disputes without litigation.
Arbitration offers alternative dispute resolution.
Negotiation finds mutually acceptable solutions.
Diplomacy prevents conflicts between nations.
Treaties establish formal agreements.
Sanctions pressure regimes to change behavior.
Aid supports nations in crisis.
Humanitarian relief saves lives in disasters.
Disaster preparedness reduces casualties and damage.
Emergency services respond to urgent situations.
Firefighters protect communities from flames.
Police maintain public order and safety.
Military defends national security.
Intelligence agencies gather strategic information.
Cyber warfare protects digital infrastructure.
Space exploration pushes beyond Earth.
Mars missions search for signs of life.
Telescopes reveal the history of the cosmos.
Black holes bend light and time.
Dark matter holds galaxies together.
Dark energy accelerates cosmic expansion.
The Big Bang created the universe.
Planets form from disks of gas and dust.
Asteroids are remnants of planetary formation.
Comets carry ice from the outer solar system.
Meteorites bring extraterrestrial material to Earth.
Auroras light up polar skies.
Eclipses block the sun or moon temporarily.
Tides are driven by gravitational forces.
Seasons result from Earth's tilted axis.
Weather systems move across continents.
Hurricanes form over warm ocean waters.
Tornadoes tear through the landscape.
Blizzards bury regions in snow.
Droughts dry up rivers and reservoirs.
Floods overwhelm roads and homes.
Wildfires consume vast areas of forest.
Earthquakes shake the ground beneath our feet.
Volcanoes erupt with molten rock and ash.
Tsunamis surge across coastlines.
Landslides bury everything in their path.
Erosion wears away mountains over millennia.
Sedimentation builds new land from debris.
Glaciers carve valleys and shape terrain.
Rivers deposit fertile soil in floodplains.
Deltas form where rivers meet the sea.
Islands rise from volcanic activity.
Atolls form around dying coral reefs.
Canyons are carved by persistent water flow.
Caves hide underground wonders.
Geysers erupt with hot water and steam.
Hot springs warm cold mountain valleys.
Deserts test the limits of endurance.
Tundras freeze under polar conditions.
Savannas support vast herds of wildlife.
Wetlands filter water and store carbon.
Marshes provide habitat for birds and fish.
Swamps teem with life and decay.
Bogs preserve ancient organic matter.
Peatlands store vast amounts of carbon.
Grasslands sway in the wind.
Woodlands shelter diverse ecosystems.
Rainforests are the lungs of the planet.
Cloud forests cling to mountain peaks.
Alpine meadows bloom above the tree line.
Desert oases sustain life in arid lands.
Coral reefs support marine biodiversity.
Mangroves protect coastlines from storms.
Seagrass meadows feed marine herbivores.
Kelp forests form underwater forests.
Open ocean is the largest habitat on Earth.
Deep sea trenches hold mysteries of the abyss.
Hydrothermal vents support unique ecosystems.
Bioluminescence lights up the dark ocean.
Whales migrate thousands of miles each year.
Dolphins communicate with complex clicks.
Sharks are ancient and apex predators.
Turtles navigate using Earth's magnetic field.
Octopuses are masters of camouflage.
Jellyfish drift with ocean currents.
Plankton form the base of marine food webs.
Fish schools move as one organism.
Coral polyps build massive reef structures.
Sea stars regenerate lost body parts.
Sponges filter water with microscopic pores.
Anemones sting with tiny venomous cells.
Starfish cling to rocky shorelines.
Crayfish scuttle along river bottoms.
Crabs scavenge along sandy beaches.
Lobsters hide in rocky crevices.
Shrimp filter plankton from seawater.
Mussels attach to rocks and shells.
Clams burrow into sandy seabeds.
Oysters build reefs that shelter other life.
Pearls form inside oyster shells.
Sponges produce compounds used in medicine.
Seaweeds absorb nutrients from seawater.
Kelp grows rapidly in cold waters.
Algae form the base of aquatic ecosystems.
Diatoms build glass-like microscopic shells.
Cyanobacteria produce oxygen through photosynthesis.
Bacteria decompose dead organic matter.
Fungi break down tough plant material.
Mushrooms spread spores on the wind.
Molds grow on decaying surfaces.
Yeasts ferment sugars into alcohol and CO2.
Viruses infect cells and hijack machinery.
Antibodies target and neutralize pathogens.
T cells coordinate immune responses.
B cells produce specific antibodies.
Macrophages engulf and digest invaders.
Dendritic cells present antigens to T cells.
Natural killer cells destroy infected cells.
Interferons inhibit viral replication.
Cytokines signal between immune cells.
Inflammation recruits healing cells to injury.
Fever raises body temperature to fight infection.
Antibiotics kill or inhibit bacterial growth.
Vaccines train the immune system in advance.
Immunity protects against future infections.
Autoimmune diseases attack the body itself.
Allergies overreact to harmless substances.
Transplant rejection occurs when immune system fights donor tissue.
Immunosuppressants prevent organ rejection.
Stem cells can become any cell type.
Tissue engineering grows replacement organs.
Gene therapy corrects defective DNA.
CRISPR edits genes with precision.
Cloning creates genetic copies of organisms.
Hybridization combines traits from different species.
Selective breeding improves crop yields.
Genetically modified organisms resist pests.
Organic farming avoids synthetic chemicals.
Permaculture designs sustainable food systems.
Aquaponics combines fish farming with hydroponics.
Vertical farming grows crops in stacked layers.
Precision agriculture uses data to optimize yields.
Drones monitor crop health from above.
Sensors measure soil moisture and nutrients.
Satellites track weather and climate patterns.
Weather forecasting saves lives and property.
Climate models predict future warming.
Carbon capture stores CO2 underground.
Carbon offsets fund renewable energy projects.
Reforestation restores degraded forests.
Afforestation plants trees where none existed.
Conservation protects endangered species.
Wildlife corridors connect fragmented habitats.
National parks preserve natural landscapes.
Marine reserves protect ocean ecosystems.
Sustainable fishing prevents overharvesting.
Recycling reduces waste and conserves materials.
Composting returns nutrients to soil.
Upcycling transforms waste into valuable products.
Circular economy eliminates waste by design.
Zero waste aims to send nothing to landfills.
Minimalism reduces consumption and clutter.
Slow living values quality over quantity.
Simplicity brings clarity and peace.
Nature restores the weary mind.
Forest bathing reduces stress and anxiety.
Gardening connects us to the seasons.
Birdwatching reveals hidden beauty.
Stargazing inspires wonder and awe.
Hiking tests body and spirit.
Climbing pushes limits of endurance.
Swimming strengthens every muscle group.
Running clears the mind and body.
Cycling explores new places efficiently.
Sailing harnesses wind for propulsion.
Surfing rides the power of ocean waves.
Skating glides on ice or wheels.
Skiing descends snow-covered slopes.
Snowboarding carves through fresh powder.
Rock climbing tests grip and balance.
Caving explores dark underground passages.
Diving reveals underwater worlds.
Snorkeling observes coral reefs up close.
Kayaking paddles through rivers and lakes.
Rafting races down rapids together.
Sailing races test speed and strategy.
Rowing builds strength and coordination.
Windsurfing combines surfing and sailing.
Kitesurfing harnesses wind and board.
Paragliding soars on thermal currents.
Skydiving freefalls through the atmosphere.
Bungee jumping tests courage and nerves.
White water rafting demands teamwork.
Survival training teaches wilderness skills.
Navigation finds direction without GPS.
Fire making creates warmth and light.
Shelter building protects from elements.
Foraging identifies edible wild plants.
Fishing provides protein from water bodies.
Hunting tracks and harvests game animals.
Trapping catches fur-bearing animals.
Butchering processes meat for consumption.
Preserving food extends shelf life.
Canning stores food in sealed containers.
Drying removes water to prevent spoilage.
Smoking flavors and preserves meat and fish.
Pickling ferments food in vinegar or brine.
Fermentation transforms sugars into alcohol and acids.
Baking turns dough into bread and pastries.
Roasting concentrates flavors through dry heat.
Grilling cooks food over open flame.
Frying crisps food in hot oil.
Boiling cooks food in bubbling water.
Steaming preserves nutrients and texture.
Searing creates a flavorful crust on meat.
Braising tenderizes tough cuts slowly.
Poaching cooks gently in barely simmering liquid.
Braising combines searing and slow cooking.
Blanching briefly cooks vegetables for freezing.
Sauteing cooks quickly in a little fat.
Stir frying tosses ingredients in hot wok.
Deep frying submerges food in hot oil.
Pan frying cooks with minimal oil.
Griddling cooks on a flat heated surface.
Broiling cooks with direct overhead heat.
Microwaving heats with electromagnetic radiation.
Slow cooking tenderizes over many hours.
Pressure cooking speeds up cooking under pressure.
Sous vide cooks at precise temperatures.
Charcoal grilling imparts smoky flavor.
Wood firing uses burning wood for heat.
Ember roasting cooks in hot coals.
Pit roasting buries food underground with hot stones.
Open fire cooking is the oldest method.
Campfire cooking brings people together.
Backpacking meals must be light and nutritious.
Dehydrated meals rehydrate with hot water.
Energy bars provide quick calories on the trail.
Trail mix combines nuts and dried fruit.
Jerky preserves meat for long journeys.
Hardtack is a dense biscuit that lasts forever.
Grape leaves wrap fillings in soft parcels.
Tortillas flatten dough into portable bread.
Flatbreads are the oldest form of bread.
Bagels boil then bake for chewy texture.
Croissants layer butter and dough for flaky layers.
Pretzels twist dough into distinctive shapes.
Focaccia tops bread with olive oil and herbs.
Ciabatta has a crunchy crust and airy crumb.
Sourdough ferments with wild yeast and bacteria.
Rye bread has a dense dark crumb.
Whole wheat bread retains the bran and germ.
Multigrain bread mixes several cereal grains.
Bran bread adds high fiber bran flakes.
Oat bread incorporates wholesome oats.
Cornbread uses ground cornmeal for base.
Pancakes flatten batter into golden rounds.
Waffles iron batter into grid patterns.
Crepes spread thin batter into delicate sheets.
Blintzes fold crepes around sweet fillings.
Dumplings wrap filling in dough pockets.
Ravioli fills pasta squares with cheese or meat.
Gyoza pan fry Japanese dumplings golden.
Dumplings steam in bamboo baskets.
Bao buns steam into fluffy white rounds.
Empanadas fill pastry with savory mixtures.
Pierogi fold dough around potato or cheese.
Spring rolls wrap vegetables in thin wrappers.
Egg rolls fry into crispy golden cylinders.
Samosas triangle pastry around spiced filling.
Pastries layer butter and dough for flaky texture.
Tarts fill pastry shells with sweet or savory fillings.
Pies encase filling in top and bottom crust.
Quiche bakes eggs and cheese in pastry shell.
Scones bake quick bread with cream and fruit.
Biscuits rise with butter pockets and steam.
Muffins combine quick bread and fruit or nuts.
Cakes layer sweet batter with frosting.
Cookies drop batter into small sweet bites.
Brownies bake dense chocolate squares.
Bars slice into rectangular treats.
Brownies and blondies differ only in cocoa.
Fudge melts chocolate and sugar into smooth candy.
Truffles roll chocolate ganache in cocoa powder.
Caramels cook sugar and cream to soft ball.
Toffee bakes butter and sugar until hard crack.
Nougat mixes egg whites sugar and nuts.
Fondant rolls into smooth sweet paste.
Marzipan shapes almond paste into sculptures.
Licorice twists black anise-flavored candy.
Gummy bears mold gelatin into bear shapes.
Jelly beans coat hard candy with sweet shell.
Lollipops spin sugar into colorful discs.
Cotton candy spins sugar into fluffy strands.
Popcorn pops kernels under heat and pressure.
Caramel corn coats popped kernels with candy.
Puffed rice expands grains into airy snacks.
Chips fry potato slices into salty crisps.
Pretzel chips bake twisted salted snacks.
Trail mix combines nuts seeds and chocolate.
Granola bakes oats honey and dried fruit.
Muesli mixes raw oats with fresh fruit.
Cereal pours milk over crunchy flakes.
Pancakes stack golden rounds with syrup.
Waffles grid batter into crispy pockets.
French toast soaks bread in egg custard.
Oatmeal simmers oats in milk or water.
Porridge thickens grains into warm comfort food.
Grits cook cornmeal into creamy Southern staple.
Polenta grinds corn into Italian porridge.
Risotto stirs Arborio rice until creamy.
Paella cooks rice with seafood and saffron.
Biryani layers spiced rice with meat and vegetables.
Fried rice tosses day-old rice with eggs and veggies.
Rice pudding simmers rice in sweetened milk.
Arancini fry stuffed risotto into golden balls.
Couscous steams tiny semolina pearls.
Pasta boils dough into many shapes.
Spaghetti twirls long thin noodles.
Penne cuts tubes at diagonal angles.
Fusilli spirals into twisted shapes.
Farfalle pinches pasta into bow ties.
Rigatoni tubes accept chunky sauces.
Lasagna layers sheets with sauce and cheese.
Ravioli fills square pasta with cheese or meat.
Tortellini loops pasta into navel shapes.
Gnocchi rolls potato dough into pillow dumplings.
Ravioli folds filling between two pasta sheets.
Cannelloni tubes accept creamy fillings.
Manicotti stuffs large tubes with cheese.
Tortelloni rolls larger filled pasta.
Agnolotti folds thin pasta over filling.
Pappardelle cuts wide flat ribbons.
Linguine thins spaghetti slightly.
Bucatini bores a hole through thick spaghetti.
Capellini creates angel hair pasta.
Orecchio cuts pasta into small ears.
Conchiglie shapes pasta into seashells.
Rotelle cuts pasta into tiny wheels.
Stelle stamps pasta into star shapes.
Anelli rings pasta into small circles.
Orzo shapes pasta into rice-like grains.
Ditalini cuts pasta into tiny tubes.
Macaroni bends tubes into elbow shapes.
Shell pasta scoops up chunky sauces.
Ziti cuts straight tubes for baked pasta.
Penne rigate adds ridges to hold sauce.
Fettuccine cuts flat wide ribbons.
Tagliatelle trims egg pasta into strips.
Pappardelle widens ribbons for hearty sauces.
Pici rolls thick hand-rolled Tuscan pasta.
Trofie twists short thick Ligurian pasta.
Pansotti folds tri-colored pasta into pockets.
Maltagliata cuts miscut pasta into irregular shapes.
Casarecce twists pasta into short spirals.
Campanelle rings pasta into little bells.
Gemelli twins two strands of pasta.
Radiatori radiates pasta into ridged spirals.
Lumache shells pasta into snail shapes.
Conchiglie grandi creates large seashell pasta.
Casarecce twists pasta into short cylinders.
Gigli ties pasta into decorative bows.
Mafalde cuts pasta into ruffled ribbons.
Paccheri tubes accept large chunky fillings.
Ziti bakes into cheesy layered dishes.
Rigate adds ridges to pasta for sauce grip.
Gnocchetti rolls small gnocchi into pillowy shapes.
Trofie twists short thick pasta strips.
Pansotti folds filling into triangular pockets.
Agnolotti platti flattens filled pasta.
Capellini al pomodoro dresses thin pasta with tomato.
Spaghetti alle vongole clams cook with linguine.
Penne arrabiata peppers up tomato sauce.
Cacio e pepe peppers Pecorino and pasta.
Carbonara creams eggs cheese and pancetta.
Amatriciana combines guanciale with tomato sauce.
Bolognese simmers meat sauce for hours.
Puttanesa olives capers anchovies tomatoes.
Marinara quick tomatoes garlic and herbs.
Alfredo creams butter and Parmesan.
Pesto blends basil pine nuts and cheese.
Aglio e olio garlic and olive oil simplicity.
Norma eggplant tomato and basil Sicilian.
Alla gricia white carbonara without egg.
Cacio e pepe simple pepper and cheese.
Gricia guanciale pecorino pepper Roman.
Tonnato tuna cream sauce Piedmontese.
Pizzaiola pizza-style tomato sauce on pasta.
Primavera spring vegetables with pasta.
Puttanesa briny Mediterranean flavors.
Arrabbiata spicy tomato kick.
Vongole clams with white wine and garlic.
Frutti di mare seafood medley with pasta.
Scampi shrimp with garlic and butter.
Linguine al nero di seppia squid ink pasta.
Spaghetti aglio olio e peperoncino garlicky spicy.
Pasta alla norma Sicilian eggplant classic.
Ravioli ricotta e spinaci cheese and spinach.
Tortellini in brodo broth floating pasta.
Lasagna alla bolognese meat and cheese layers.
Cannelloni spinaci e ricotta spinach filling.
Rigatoni alla vodka creamy tomato vodka sauce.
Penne all arrabbiata spicy tomato perfection.
Fettuccine alfredo creamy Parmesan indulgence.
Linguine alle vongole clamy garlic bliss.
Gnocchi al pesto basil pine nut delight.
Ravioli al sugo filled pasta in tomato sauce.
Spaghetti carbonara egg cheese pancetta creaminess.
Pappardelle al cinghiale wild boar ragout.
Orecchiette alle cime di rapa broccoli rabe.
Trofie al pesto Ligurian twist and basil.
Gnocchetti sardi Sardinian small gnocchi.
Paccheri al ragù large tubes meaty sauce.
Mafalde al forno baked ruffled pasta.
Gigli in brodo bow-tie pasta in broth.
Campanelle al funghi mushroom bell pasta.
Gemelli al pomodoro twisted pasta tomato sauce.
Radiatori alle salsicce ridged spiral sausage.
Lumache alla livornese snail-shaped pasta Livorno.
Conchiglie grandi al pesto large shell pesto.
Casarecce al ragù twisted cylinder meat sauce.
Maltagliata in brodo miscut pasta chicken broth.
Pappardelle al tartufi truffle wide ribbon pasta.
Fettuccine al salmone smoked salmon cream pasta.
Tagliatelle al prosciutto prosciutto egg pasta.
Pici cinghiale wild boar hand-rolled Tuscan.
Trofie al pomodoro twisted pasta tomato simple.
Pansotti al pesto genovese folded pocket basil.
Agnolotti del plin pinched filled Piedmont pasta.
Capellini al limoncello lemon thin pasta light.
Spaghetti alle cozze mussels with spaghetti.
Penne alla norma eggplant tomato penne.
Rigatoni alla puttanesa briny ridged tube pasta.
Farfalle alfredo bow tie creamy cheese.
Conchiglie al frutt di mare shell seafood pasta.
Rotelle al pesto wheel pasta basil delight.
Stelle al pomodoro star pasta tomato simple.
Anelli al formaggio ring pasta cheese dip.
Orzo al limone rice-shaped pasta lemon.
Ditalini in zuppa tiny tube pasta soup.
Macaroni al forno baked elbow cheese pasta.
Shell alfredo creamy shell pasta cheese.
Ziti alla vodka ridged tube vodka tomato.
Penne rigate alla bolognese ridged penne meat.
Fettuccine alla carbonara ribbon egg pancetta.
Tagliatelle al ragù wide ribbon meat sauce.
Pappardelle al wild mushroom wide ribbon forest.
Linguine al scampi shrimp garlic linguine.
Bucatini all amatriciana tube guanciale tomato.
Capellini al pomodoro angel hair tomato basil.
Orecchio alle cime ear pasta broccoli rabe.
Conchiglie grandi al mare large shell seafood.
Casarecce al ragù di agnello twisted lamb sauce.
Gigli in bianco bow tie pasta white sauce.
Mafalde al forno baked ruffled ribbon pasta.
Paccheri al tonno large tube tuna sauce.
Gemelli al pesto genovese twin basil pesto.
Radiatori al ragù di salsicce ridged spiral sausage.
Lumache al sugo snail pasta tomato sauce.
Campanelle al funghi porcini bell pasta porcini.
Stelle al limoncello star pasta lemon zest.
Anelli al parmigiano ring pasta Parmesan butter.
Orzo in zuppa di pesce rice-shaped seafood soup.
Ditalini in minestrone tiny tube vegetable soup.
Macaroni e formaggio baked elbow cheese comfort.
Shell alfredo cream sauce shell pasta classic.
Ziti alla marinara tube pasta quick tomato.
Penne alla vodka e panna creamy spicy penne.
Rigatoni al cinghiale wild boar ridged tube.
Fettuccine al tartufo black truffle ribbon pasta.
Tagliatelle al burro e salvia butter sage ribbon.
Pappardelle al ragù di selvaggina game wide ribbon.
Linguine ai frutti di mare seafood linguine classic.
Bucatini all'amatriciana guanciale tube classic Roman.
Capellini allo zafferano saffron angel hair delicate.
Orecchiette al broccoli rabe ear pasta bitter sweet.
Conchiglie grandi al ragù large shell meat sauce.
Casarecce al pomodorino twisted cherry tomato pasta.
Gigli al ragù di carne bow tie meat sauce.
Mafalde alle verdure ruffled vegetable baked pasta.
Paccheri al ragù di mare large tube seafood sauce.
Gemelli al sugo di pomodoro twin tomato pasta.
Radiatori al ragù di carne ridged spiral meat.
Lumache al ragù di maiale snail pork sauce.
Campanelle al tartufo nero truffle bell pasta.
Stelle al pesto di basilico star basil pesto.
Anelli al burro e parmigiano ring butter Parmesan.
Orzo al ragù di pollo rice-shaped chicken sauce.
Ditalini in brodo di pollo tiny tube chicken broth.
Macaroni al ragù di carne baked elbow meat sauce.
Shell al sugo di tonno shell tuna tomato sauce.
Ziti alla marinara e mozzarella tube tomato cheese.
Penne alla carbonara creamy egg penne classic.
Rigatoni al ragù di agnello lamb ridged tube.
Fettuccine al salmone affumicato smoked salmon ribbon.
Tagliatelle al ragù di cinghiale wild boar ribbon.
Pappardelle al tartufo bianco white truffle wide.
Linguine ai gamberi rosse red shrimp linguine.
Bucatini all'amatriciana guanciale tube Roman classic.
Capellini al pomodorini cherry tomato angel hair.
Orecchiette alle cime di rapa broccoli rabe classic.
Conchiglie grandi ai frutti di mare large shell seafood.
Casarecce al ragù di salsicce twisted sausage sauce.
Gigli al ragù di vitello veal bow tie pasta.
Mafalde al pesto di pistacchio pistachio ruffled.
Paccheri al ragù di tonno large tube tuna sauce.
Gemelli al sugo di verdure twin vegetable sauce.
Radiatori al ragù di vitello ridged spiral veal.
Lumache al ragù di cinghiale snail wild boar.
Campanelle al ragù di salsicce bell sausage sauce.
Stelle al pesto di rucola arugula star pesto.
Anelli al ragù di maiale ring pork sauce.
Orzo al ragù di coniglio rice-shaped rabbit sauce.
Ditalini in zuppa di lenticchie tiny tube lentil soup.
Macaroni al ragù di mare baked elbow seafood sauce.
Shell al pesto di basilico shell basil pesto.
Ziti al ragù di carne tube meat sauce baked.
Penne alla norma eggplant tomato penne Sicilian.
Rigatoni al ragù di agnello lamb ridged tube.
Fettuccine al ragù di cinghiale wild boar ribbon.
Tagliatelle al tartufo nero black truffle ribbon.
Pappardelle al ragù di selvaggina game wide ribbon.
Linguine ai frutti di mare mixed seafood linguine.
Bucatini all'amatriciana guanciale tube Roman classic.
Capellini allo zafferano saffron angel hair delicate.
Orecchiette al broccoli rabe ear pasta classic Pugliese.
Conchiglie grandi al ragù di mare large shell seafood.
Casarecce al ragù di salsicce twisted sausage sauce.
Gigli al ragù di vitello veal bow tie pasta.
Mafalde al pesto di pistacchio pistachio ruffled.
Paccheri al ragù di tonno large tube tuna sauce.
Gemelli al sugo di verdure twin vegetable sauce.
Radiatori al ragù di vitello ridged spiral veal.
Lumache al ragù di cinghiale snail wild boar.
Campanelle al tartufo nero truffle bell pasta.
Stelle al pesto di rucola arugula star pesto.
Anelli al ragù di maiale ring pork sauce.
Orzo al ragù di coniglio rice-shaped rabbit sauce.
Ditalini in zuppa di lenticchie tiny tube lentil soup.
Macaroni al ragù di mare baked elbow seafood sauce.
Shell al pesto di basilico shell basil pesto.
Ziti al ragù di carne tube meat sauce baked.
"""

def ensure_corpus(data_dir: str, extra_dirs: list = None) -> list[str]:
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    text_file = data_path / "tinystories.txt"

    if text_file.exists():
        print(f"Using existing corpus: {text_file} ({os.path.getsize(text_file) / (1024*1024):.1f} MB)")
    else:
        print(f"Corpus not found at {text_file}")
        print(f"Run: python init_training.py --download --force")
        print(f"Falling back to built-in corpus.")
        text_file = data_path / "wiki_train.txt"
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(BUILTIN_CORPUS)

    all_sentences = []
    text_files = [text_file]

    if extra_dirs:
        for extra in extra_dirs:
            extra_path = Path(extra)
            if extra_path.exists():
                for fn in sorted(extra_path.glob("*.txt")):
                    if fn not in text_files:
                        text_files.append(fn)
                        print(f"  Found extra corpus: {fn} ({fn.stat().st_size / (1024*1024):.0f} MB)")

    for tf in text_files:
        print(f"  Reading {tf} ...")
        with open(tf, "r", encoding="utf-8") as f:
            raw = f.read()
        sentences = [s.strip() for s in raw.split("\n") if s.strip()]
        all_sentences.extend(sentences)

    print(f"Loaded {len(all_sentences)} total sentences from {len(text_files)} files")
    return all_sentences


# =============================================================================
# 3. MODEL: Causal GPT (GPT-mini2 architecture)
# =============================================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        n_embd = config["n_head"] * config["head_dim"]
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(0.0)
        self.resid_drop = nn.Dropout(0.0)
        self.n_head = config["n_head"]
        self.head_dim = config["head_dim"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        mask = torch.tril(torch.ones(T, T, device=x.device))
        att = att.masked_fill(mask == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.c_proj(y))
        return y


class Block(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        n_embd = config["n_head"] * config["head_dim"]
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTMini(nn.Module):
    def __init__(self, config: dict, vocab_size: int):
        super().__init__()
        self.n_embd = config["n_head"] * config["head_dim"]
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(vocab_size, self.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config["n_layer"])]),
                ln_f=nn.LayerNorm(self.n_embd),
            )
        )
        self.register_buffer(
            "wpe", torch.zeros(1, config["seq_length"], self.n_embd)
        )
        self.lm_head = nn.Linear(self.n_embd, vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"number of parameters: {n_params/1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wpe.numel()
        return n_params

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        b, t = idx.size()
        assert t <= self.wpe.size(1), f"Cannot forward, seq_length exhausted ({t} > {self.wpe.size(1)})"
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.wpe[:, :t, :]
        x = tok_emb + pos_emb
        if self.training:
            from torch.utils.checkpoint import checkpoint
            for block in self.transformer.h:
                x = checkpoint(block, x, use_reentrant=True)
        else:
            for block in self.transformer.h:
                x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), targets.view(-1).long())
        return logits, loss


# =============================================================================
# 4. GENERATION
# =============================================================================
def generate_text(model: GPTMini, tokenizer: WordTokenizer, prompt: str, max_new_tokens: int = 50, temperature: float = 0.8, device: str = "cpu") -> str:
    model.eval()
    tokens = tokenizer.encode(prompt)
    tokens = tokens[-model.wpe.size(1):]
    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_tensor)[0]
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            if next_token == tokenizer.word2idx.get("<eos>", 2):
                break
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], dtype=torch.long, device=device)], dim=1)

    generated_tokens = input_tensor.squeeze().tolist()
    return tokenizer.decode(generated_tokens)


# =============================================================================
# 5. HASHES
# =============================================================================
def get_vocab_hash(vocab_cfg: dict, data_dirs: list) -> str:
    """Hash tokenizer config + data source file metadata (name, size, mtime).
    The result is the single source of truth for vocab/data cache naming."""
    h = hashlib.sha256()
    # Tokenizer params
    h.update(json.dumps({
        "max_vocab_size": vocab_cfg.get("max_vocab_size", 32768),
        "max_word_len": vocab_cfg.get("max_word_len", 20)
    }, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    # Data source file metadata
    file_meta = []
    for d in data_dirs:
        dp = Path(d)
        if dp.exists():
            for fp in sorted(dp.glob("*.txt")):
                st = fp.stat()
                file_meta.append({"n": fp.name, "s": st.st_size, "t": st.st_mtime})
    file_meta.sort(key=lambda x: x["n"])
    h.update(json.dumps(file_meta, separators=(",", ":")).encode("utf-8"))
    return h.hexdigest()[:16]


def get_model_hash(model, vocab_hash: str = None) -> str:
    """Derive a deterministic hash from the model's actual tensor-defining
    attributes + vocabulary identity.  Including vocab_hash ensures
    checkpoints are never shared between different vocabularies, which
    would make embedding weights meaningless."""
    m = model.module if hasattr(model, "module") else model
    h = hashlib.sha256()
    h.update(json.dumps({
        "vocab_size": int(m.transformer.wte.num_embeddings),
        "n_embd": int(m.n_embd),
        "n_layer": len(m.transformer.h),
        "n_head": int(m.transformer.h[0].attn.n_head),
        "head_dim": int(m.transformer.h[0].attn.head_dim),
        "seq_length": int(m.wpe.size(1)),
    }, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    if vocab_hash:
        h.update(vocab_hash.encode("utf-8"))
    return h.hexdigest()[:16]


# =============================================================================
# 6. CHECKPOINTING
# =============================================================================
# Layout: <ckpt_dir>/<model_hash>/          ← latest checkpoint + config (once)
#         <ckpt_dir>/<model_hash>/1         ← every 10th epoch
#         <ckpt_dir>/<model_hash>/2         ← every 100th epoch
#         <ckpt_dir>/<model_hash>/3         ← every 1000th epoch
#         <ckpt_dir>/<model_hash>/4         ← every 10000th epoch
#         ...
#         <ckpt_dir>/<cfg_hash>/15         ← every 1000000000000000th epoch


def _write_tier(ckpt_dir: str, cfg_hash: str, tier: int, epoch: int, loss: float, config: dict | None, model: GPTMini, extra: dict | None = None):
    if tier == 0:
        d = Path(ckpt_dir) / cfg_hash
    else:
        d = Path(ckpt_dir) / cfg_hash / str(tier)
    d.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), d / "model.pth")

    # Resume metadata as compact JSON
    meta = {"epoch": epoch, "loss": round(loss, 6), "config_hash": cfg_hash}
    if extra:
        meta.update(extra)
    with open(d / "resume.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, separators=(",", ":"))

    # Config only in base tier, only once
    if tier == 0 and config is not None:
        cfg_out = dict(config, _hash=cfg_hash)
        with open(d / "config.json", "w", encoding="utf-8") as f:
            json.dump(cfg_out, f, indent=2)


def _tiers_for_epoch(epoch: int) -> list[int]:
    tiers = []
    t = 1
    threshold = 10
    while threshold <= 10**15:
        if epoch % threshold == 0:
            tiers.append(t)
        t += 1
        threshold *= 10
    return tiers


# =============================================================================
# 5b. CHECKPOINT BACKUP / RESTORE
# =============================================================================
def _backup_checkpoint(ckpt_dir: Path):
    """Before overwriting, back up model.pth → model.pth.bak (and tiers)."""
    pth = ckpt_dir / "model.pth"
    if pth.exists():
        bak = ckpt_dir / "model.pth.bak"
        import shutil
        shutil.copy2(str(pth), str(bak))


def _try_restore_checkpoint(ckpt_dir: Path):
    """If model.pth is corrupt, try restoring from .bak; return True if restored."""
    pth = ckpt_dir / "model.pth"
    bak = ckpt_dir / "model.pth.bak"
    if bak.exists():
        import shutil
        shutil.copy2(str(bak), str(pth))
        return True
    return False


def _cleanup_corrupt_checkpoint(ckpt_dir: Path):
    """Delete corrupt model.pth and any .bak."""
    pth = ckpt_dir / "model.pth"
    bak = ckpt_dir / "model.pth.bak"
    if pth.exists():
        pth.unlink()
    if bak.exists():
        bak.unlink()


def save_checkpoint(epoch: int, loss: float, config: dict, cfg_hash: str, model: GPTMini, ckpt_dir: str, extra: dict | None = None):
    base = Path(ckpt_dir) / cfg_hash
    _backup_checkpoint(base)
    needs_config = not (base / "config.json").exists()
    _write_tier(ckpt_dir, cfg_hash, 0, epoch, loss, config if needs_config else None, model, extra)
    tiers = _tiers_for_epoch(epoch)
    if tiers:
        for t in tiers:
            _write_tier(ckpt_dir, cfg_hash, t, epoch, loss, None, model, extra)
        print(f"  -> Saved checkpoint at epoch {epoch} (tiers {', '.join(map(str, tiers))})")


def find_latest_checkpoint(ckpt_dir: str, expected_hash: str):
    base = Path(ckpt_dir) / expected_hash
    meta = base / "resume.json"
    if not meta.exists():
        return None
    info = json.loads(meta.read_text())
    ep = int(info.get("epoch", 0))
    loss = float(info.get("loss", 0))
    return (ep, info, base)


# =============================================================================
# 6. TRAINING
# =============================================================================
def setup_ddp():
    if dist.is_initialized():
        return int(os.environ.get("LOCAL_RANK", 0))
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Force IPv4 on Windows (hostname resolves to IPv6 -> error 10049)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        print(f"  DDP initialized: rank={dist.get_rank()}, local_rank={local_rank}, world_size={dist.get_world_size()}")
        return local_rank
    return 0  # Single process


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def _log_error(err_file, msg):
    if err_file is None:
        return
    import traceback
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    err_file.write(f"{ts}\t{msg}\n")
    err_file.flush()


def _write_status(status_file, epoch, global_batch, loss, training_samples, seq_length, training_start_time):
    if status_file is None:
        return
    elapsed = time.time() - training_start_time
    tok_per_sec = training_samples * seq_length / elapsed
    batch_per_sec = global_batch / elapsed
    line = f"{time.strftime('%H:%M:%S')}\t{epoch}\t{global_batch}\t{loss:.4f}\t{tok_per_sec:.0f}\t{batch_per_sec:.1f}\t{training_samples}\n"
    status_file.write(line)
    status_file.flush()


def train():
    print(f"Python: {sys.executable}")

    # Check if running in DDP mode
    in_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    local_rank = setup_ddp() if in_ddp else 0

    # Load config - supports both combined and split formats
    config_path = "gpt_mini3.json"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    with open(config_path, "r") as f:
        full_config = json.load(f)

    # Extract model_config and training_config from combined format
    if "model" in full_config and "training" in full_config:
        model_cfg = dict(full_config["model"])
        train_cfg = full_config["training"]
        paths = full_config.get("paths", {})
    else:
        # Legacy format: model/tok/tr directly
        model_cfg = dict(full_config.get("model", full_config))
        train_cfg = full_config.get("training", {})
        paths = full_config.get("paths", {})

    vocab_cfg = model_cfg.pop("vocab") if "vocab" in model_cfg else model_cfg.pop("tokenizer", {})

    DEVICE = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = (local_rank == 0)
    print(f"Device: {DEVICE} (rank={local_rank})")
    if is_main:
        print(f"CUDA: {torch.cuda.is_available()} (GPU: {torch.cuda.get_device_name(DEVICE.index)})")

    cache_dir = Path(paths.get("cache_dir", "E:\\training\\cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Collect all data directories for vocab hash
    data_dirs = [paths["data_dir"]]
    if "extra_data_dirs" in paths:
        data_dirs.extend(paths["extra_data_dirs"])

    vocab_hash = get_vocab_hash(vocab_cfg, data_dirs)
    if is_main:
        print(f"Vocab hash: {vocab_hash}")

    vocab_cache = cache_dir / f"vocab-{vocab_hash}.json"

    # Tokenizer
    tokenizer = WordTokenizer(max_vocab_size=vocab_cfg.get("max_vocab_size", 32768), max_word_len=vocab_cfg.get("max_word_len", 20))

    # Corpus hash for data cache invalidation
    def corpus_hash(data_dirs):
        """Hash all files in data_dirs for data cache invalidation."""
        h = hashlib.sha256()
        for data_dir in data_dirs:
            for root, _, files in os.walk(data_dir):
                for fn in sorted(files):
                    fp = Path(root) / fn
                    with open(fp, "rb") as f:
                        for chunk in iter(lambda: f.read(8192), b""):
                            h.update(chunk)
        return h.hexdigest()[:16]

    corpus_h = corpus_hash(data_dirs)
    data_cache = cache_dir / f"data-{vocab_hash}-{corpus_h}.npy"

    sentences = []
    if vocab_cache.exists():
        if is_main:
            print(f"Loading cached vocab from {vocab_cache}", flush=True)
        tokenizer.load(str(vocab_cache))
    else:
        if is_main:
            sentences = ensure_corpus(paths["data_dir"], paths.get("extra_data_dirs", []))
            tokenizer.build_vocab(sentences)
            tokenizer.save(str(vocab_cache))
            print(f"Vocab cached to {vocab_cache}", flush=True)

    if data_cache.exists() and data_cache.stat().st_size > 1_000_000_000:
        if is_main:
            print(f"Loading cached dataset ({data_cache.stat().st_size // 1_000_000_000}GB)...", flush=True)
        sentences = []
    elif sentences:
        pass  # sentences already loaded from vocab build
    else:
        if is_main:
            sentences = ensure_corpus(paths["data_dir"], paths.get("extra_data_dirs", []))

    # Dataset
    dataset = WordDataset(sentences, tokenizer, model_cfg["seq_length"], cache_file=str(data_cache))
    sampler = DistributedSampler(dataset, shuffle=True) if dist.is_initialized() else None
    dataloader = DataLoader(dataset, batch_size=train_cfg["batch_size"], sampler=sampler, drop_last=True)
    if is_main:
        print(f"Dataset: {len(dataset)} samples", flush=True)

    # Model
    model = GPTMini(model_cfg, tokenizer.vocab_size).to(DEVICE)

    # DDP wrap
    if dist.is_initialized():
        find_unused_parameters = False
        for name, param in model.named_parameters():
            if param.requires_grad:
                find_unused_parameters = True
                break
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=find_unused_parameters)

    # Get unwrapped model for parameter access
    unwrapped_model = model.module if hasattr(model, "module") else model

    # Canonical checkpoint hash — derived from actual model tensor dims
    ckpt_hash = get_model_hash(model, vocab_hash)
    if is_main:
        print(f"Checkpoint hash: {ckpt_hash}", flush=True)

    # Resume check
    start_epoch = 0
    global_batch = 0
    ckpt_state = None
    if is_main:
        ckpt = find_latest_checkpoint(paths["checkpoint_dir"], ckpt_hash)
        if ckpt:
            ep, info, ckpt_path = ckpt
            global_batch = int(info.get("global_batch", 0))
            print(f"Resuming from {ckpt_path} (epoch {ep}, loss {info['loss']:.6f}, global_batch {global_batch})", flush=True)
            ckpt_state = torch.load(ckpt_path / "model.pth", map_location=DEVICE)
            start_epoch = ep
    else:
        ckpt = None

    if dist.is_initialized():
        # Broadcast resume info from rank 0 to all ranks
        resume_data = [start_epoch, global_batch]
        dist.broadcast_object_list(resume_data, src=0)
        start_epoch, global_batch = resume_data

    # Load checkpointed weights
    if ckpt_state is not None:
        unwrapped_model.load_state_dict(ckpt_state)

    optimizer = torch.optim.Adam(unwrapped_model.parameters(), lr=train_cfg["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    ckpt_interval = train_cfg.get("checkpoint_interval", 0)
    ckpt_every_min = train_cfg.get("checkpoint_every_min", 0)
    grad_accum = train_cfg.get("gradient_accumulation_steps", 1)

    debug_one_step = os.environ.get("DEBUG_ONE_STEP", "0") == "1"

    # Combined config for checkpoints (model + training)
    combined_config = {"model": model_cfg, "training": train_cfg, "paths": paths}
    combined_config["model"]["vocab"] = vocab_cfg

    log_file = None
    err_file = None
    training_start_time = None
    if is_main:
        ckpt_dir = Path(paths["checkpoint_dir"])
        ckpt_base = ckpt_dir / ckpt_hash
        ckpt_base.mkdir(parents=True, exist_ok=True)
        log_file = open(ckpt_base / "checkpoint_status.txt", "a", encoding="utf-8")
        err_file = open(ckpt_base / "errors.log", "w", encoding="utf-8")
        training_start_time = time.time()
        _ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"Start time: {_ts}", flush=True)
        print("Starting training..." + (" [DEBUG_ONE_STEP]" if debug_one_step else ""), flush=True)

    last_ckpt_time = time.time()
    epoch_start_time = None
    num_batches = 0
    total_loss = 0.0
    training_samples = 0
    for epoch in range(start_epoch + 1, train_cfg["epochs"] + 1):
        if dist.is_initialized():
            try:
                sampler.set_epoch(epoch)
            except Exception as e:
                _log_error(err_file, f"sampler.set_epoch({epoch}): {e}")
        model.train()
        epoch_start_time = time.time()
        for batch_idx, (x, y) in enumerate(dataloader):
            if debug_one_step and (batch_idx > 0 or global_batch > 0):
                if is_main:
                    print(f"  [DEBUG] breaking after 1 batch", flush=True)
                break
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            global_batch += 1
            training_samples += x.size(0)
            should_ckpt = False
            if ckpt_interval > 0 and global_batch % ckpt_interval == 0:
                should_ckpt = True
            if ckpt_every_min > 0 and (time.time() - last_ckpt_time) >= ckpt_every_min * 60:
                should_ckpt = True
            if should_ckpt:
                avg = total_loss / max(1, num_batches)
                try:
                    if is_main:
                        _write_status(log_file, epoch, global_batch, avg, training_samples, model_cfg["seq_length"], training_start_time)
                        save_checkpoint(epoch, avg, combined_config, ckpt_hash, unwrapped_model, paths["checkpoint_dir"],
                                        extra={"global_batch": global_batch, "batch_size": train_cfg["batch_size"],
                                               "seq_length": model_cfg["seq_length"], "training_samples": training_samples})
                    if dist.is_initialized():
                        dist.barrier()
                except Exception as e:
                    _log_error(err_file, f"checkpoint batch {global_batch}: {e}")
                last_ckpt_time = time.time()
                num_batches = 0
                total_loss = 0.0
        avg_loss = total_loss / max(1, num_batches)
        try:
            scheduler.step()
        except Exception as e:
            _log_error(err_file, f"scheduler.step epoch {epoch}: {e}")

        if epoch % train_cfg["checkpoint_every"] == 0:
            try:
                if is_main:
                    _write_status(log_file, epoch, global_batch, avg_loss, training_samples, model_cfg["seq_length"], training_start_time)
                    save_checkpoint(epoch, avg_loss, combined_config, ckpt_hash, unwrapped_model, paths["checkpoint_dir"],
                                    extra={"global_batch": global_batch, "batch_size": train_cfg["batch_size"],
                                           "seq_length": model_cfg["seq_length"], "training_samples": training_samples})
                if dist.is_initialized():
                    dist.barrier()
            except Exception as e:
                _log_error(err_file, f"epoch checkpoint {epoch}: {e}")

    if is_main:
        if log_file:
            log_file.close()
        if err_file:
            err_file.close()

    if dist.is_initialized():
        cleanup_ddp()


if __name__ == "__main__":
    train()

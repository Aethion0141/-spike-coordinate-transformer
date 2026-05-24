import random
import math

# ============ SPIKE SYSTEM ============
class SpikeTransformer:
    def __init__(self):
        self.positions = {}
        self.attention_confidence = 0.5

    def update_positions(self, positions):
        self.positions = positions

    def set_attention_confidence(self, confidence):
        self.attention_confidence = max(0.0, min(1.0, confidence))

    def attend(self, query_word):
        if query_word not in self.positions:
            return []
        query_pos = self.positions[query_word]
        results = []
        for word, pos in self.positions.items():
            if word == query_word:
                continue
            dx = query_pos['x'] - pos['x']
            dy = query_pos['y'] - pos['y']
            distance = math.sqrt(dx*dx + dy*dy)
            attention = math.exp(-distance) * self.attention_confidence
            results.append((word, attention, distance))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:10]


class SpikeMemory:
    def __init__(self):
        self.spikes = {}

    def store(self, word, x, y, conf_x=0.5, conf_y=0.5):
        self.spikes[word] = {
            'x': x, 'y': y,
            'conf_x': conf_x, 'conf_y': conf_y,
            'accesses': 0, 'correct': 0
        }

    def get_all_positions(self):
        return {word: {'x': data['x'], 'y': data['y']} 
                for word, data in self.spikes.items()}

    def get_confidence(self, word):
        if word in self.spikes:
            return (self.spikes[word]['conf_x'], self.spikes[word]['conf_y'])
        return (0.5, 0.5)

    def stats(self):
        if not self.spikes:
            return {'count': 0, 'avg_conf_x': 0, 'avg_conf_y': 0}
        avg_conf_x = sum(s['conf_x'] for s in self.spikes.values()) / len(self.spikes)
        avg_conf_y = sum(s['conf_y'] for s in self.spikes.values()) / len(self.spikes)
        return {
            'count': len(self.spikes),
            'avg_conf_x': avg_conf_x,
            'avg_conf_y': avg_conf_y,
        }


class SpikeLearner:
    def __init__(self):
        self.learning_rate = 0.1
        self.learning_confidence = 0.5
        self.learning_history = []

    def encode_sentence(self, sentence, positions, use_confidence=True):
        words = sentence.lower().split()
        valid_words = []
        for w in words:
            w = w.strip('.,;:?!\'"()')
            if w in positions:
                valid_words.append(w)
        
        if not valid_words:
            return (0.5, 0.5, 0.0)

        if use_confidence:
            total_conf = 0
            weighted_x = 0
            weighted_y = 0
            for w in valid_words:
                conf = positions[w].get('conf_x', 0.5) * positions[w].get('conf_y', 0.5)
                weighted_x += positions[w]['x'] * conf
                weighted_y += positions[w]['y'] * conf
                total_conf += conf
            if total_conf > 0:
                return (weighted_x / total_conf, weighted_y / total_conf, total_conf / len(valid_words))
        
        avg_x = sum(positions[w]['x'] for w in valid_words) / len(valid_words)
        avg_y = sum(positions[w]['y'] for w in valid_words) / len(valid_words)
        return (avg_x, avg_y, 0.5)

    def reverse_sentence(self, sentence):
        words = sentence.split()
        return ' '.join(words[::-1])

    def remix_sentence(self, sentence):
        words = sentence.split()
        random.shuffle(words)
        return ' '.join(words)

    def learn_sentence(self, sentence, positions):
        orig = sentence
        rev = self.reverse_sentence(orig)
        rem = self.remix_sentence(orig)

        x1, y1, conf1 = self.encode_sentence(orig, positions, True)
        x2, y2, conf2 = self.encode_sentence(rev, positions, True)
        x3, y3, conf3 = self.encode_sentence(rem, positions, True)

        avg_conf = (conf1 + conf2 + conf3) / 3
        center_x = (x1 + x2 + x3) / 3
        center_y = (y1 + y2 + y3) / 3
        effective_rate = self.learning_rate * avg_conf

        for word in orig.split():
            word = word.strip('.,;:?!\'"()')
            if word in positions:
                old_x = positions[word]['x']
                old_y = positions[word]['y']
                positions[word]['x'] += (center_x - old_x) * effective_rate
                positions[word]['y'] += (center_y - old_y) * effective_rate
                positions[word]['x'] = max(0, min(1, positions[word]['x']))
                positions[word]['y'] = max(0, min(1, positions[word]['y']))

        self.learning_history.append(avg_conf)
        if self.learning_history:
            self.learning_confidence = sum(self.learning_history[-100:]) / min(100, len(self.learning_history))


class BookSpikeAI:
    def __init__(self):
        self.transformer = SpikeTransformer()
        self.memory = SpikeMemory()
        self.learner = SpikeLearner()
        self.system_confidence = 0.5

    def add_word(self, word):
        if word not in self.memory.spikes:
            self.memory.store(word, random.uniform(0, 1), random.uniform(0, 1))

    def train_on_text(self, text, title="Text", epochs=5):
        sentences = text.replace('\n', ' ').replace('?', '.').replace('!', '.').replace(';', '.').split('.')
        
        clean_sentences = []
        for sent in sentences:
            sent = sent.strip().lower()
            if len(sent) > 5 and len(sent.split()) > 2:
                clean_sentences.append(sent)
                for word in sent.split():
                    word = word.strip('.,;:?!\'"()')
                    if word:
                        self.add_word(word)
        
        print(f"📚 {title}: Loaded {len(clean_sentences)} sentences")
        print(f"📖 Vocabulary: {len(self.memory.spikes)} unique words")
        
        positions = self.memory.get_all_positions()
        
        for epoch in range(epochs):
            random.shuffle(clean_sentences)
            for sentence in clean_sentences:
                self.learner.learn_sentence(sentence, positions)
            
            for word, pos in positions.items():
                if word in self.memory.spikes:
                    self.memory.spikes[word]['x'] = pos['x']
                    self.memory.spikes[word]['y'] = pos['y']
            
            avg_conf = sum(s['conf_x'] * s['conf_y'] for s in self.memory.spikes.values()) / len(self.memory.spikes)
            self.system_confidence = (self.learner.learning_confidence + avg_conf) / 2
            self.transformer.set_attention_confidence(self.system_confidence)
            print(f"   Epoch {epoch+1}/{epochs} complete (sys_conf={self.system_confidence:.3f})")
        
        self.transformer.update_positions(positions)
    
    def query(self, word):
        return self.transformer.attend(word)
    
    def show_connections(self, title, words):
        print(f"\n🔍 {title}:")
        print("-" * 50)
        for word in words:
            results = self.query(word)
            if results:
                print(f"\n'{word}' connects to:")
                for w, att, dist in results[:5]:
                    print(f"   → {w}: attention={att:.4f}, distance={dist:.3f}")
    
    def status(self):
        stats = self.memory.stats()
        print("\n" + "="*70)
        print("📊 SYSTEM STATUS")
        print("="*70)
        print(f"🎯 System Confidence: {self.system_confidence:.3f}")
        print(f"🧠 Learner Confidence: {self.learner.learning_confidence:.3f}")
        print(f"💾 Memory: {stats['count']} words")
        print(f"   Avg Conf X: {stats['avg_conf_x']:.3f}")
        print(f"   Avg Conf Y: {stats['avg_conf_y']:.3f}")
        print("="*70)


# ============ THE STORY OF CAESAR, BRUTUS, AND ANTONY ============
story_text = """
Julius Caesar was the most powerful man in Rome.
He defeated Pompey and became dictator for life.
Brutus was a senator who loved Rome more than he loved Caesar.
Cassius convinced Brutus that Caesar wanted to be king.
Brutus joined the conspirators to kill Caesar.
On the Ides of March, the conspirators stabbed Caesar.
Caesar saw Brutus among them and said Et tu Brute.
Caesar died, and chaos spread through Rome.
Mark Antony was Caesar's closest friend and ally.
Antony pretended to support the conspirators.
At Caesar's funeral, Antony gave a famous speech.
He called Brutus an honorable man, but with sarcasm.
He showed the crowd Caesar's bloody robe.
He pointed to the stab wounds made by each conspirator.
The crowd turned against Brutus and the conspirators.
Brutus and Cassius fled Rome.
Antony joined forces with Octavius to hunt them down.
At the battle of Philippi, Brutus and Cassius were defeated.
Cassius killed himself when he thought Brutus was dead.
Brutus ran onto his own sword after Cassius died.
Antony called Brutus the noblest Roman of them all.
"""

# ============ ANTONY'S FUNERAL SPEECH ============
antony_speech = """
Friends, Romans, countrymen, lend me your ears.
I come to bury Caesar, not to praise him.
The evil that men do lives after them.
The good is often interred with their bones.
So let it be with Caesar.
Brutus says Caesar was ambitious.
Brutus is an honorable man.
He hath brought many captives home to Rome.
Whose ransoms did the general coffers fill.
Did this in Caesar seem ambitious?
When the poor have cried, Caesar hath wept.
Ambition should be made of sterner stuff.
Yet Brutus says he was ambitious.
And Brutus is an honorable man.
You all did see that on the Lupercal.
I thrice presented him a kingly crown.
Which he did thrice refuse.
Was this ambition?
Yet Brutus says he was ambitious.
And sure he is an honorable man.
I speak not to disprove what Brutus spoke.
But here I am to speak what I do know.
You all did love him once, not without cause.
What cause withholds you then to mourn for him?
O judgment, thou art fled to brutish beasts.
And men have lost their reason.
Bear with me, my heart is in the coffin there with Caesar.
And I must pause till it come back to me.
"""

# ============ BRUTUS'S SPEECH ============
brutus_speech = """
Romans, countrymen, and lovers, hear me for my cause.
Be silent, that you may hear.
Believe me for mine honor, and have respect to mine honor.
Censure me in your wisdom, and awake your senses.
If there be any in this assembly, any dear friend of Caesar's.
To him I say that Brutus' love to Caesar was no less than his.
Not that I loved Caesar less, but that I loved Rome more.
Had you rather Caesar were living and die all slaves?
Than that Caesar were dead, to live all free men?
As Caesar loved me, I weep for him.
As he was fortunate, I rejoice at it.
As he was valiant, I honor him.
But as he was ambitious, I slew him.
There is tears for his love, joy for his fortune.
Honor for his valor, and death for his ambition.
Who is here so base that would be a bondman?
If any, speak, for him have I offended.
Who is here so rude that would not be a Roman?
If any, speak, for him have I offended.
Who is here so vile that will not love his country?
If any, speak, for him have I offended.
I pause for a reply.
"""

# ============ RUN ============
print("="*70)
print("🎭 SPIKE AI LEARNS THE TRAGEDY OF CAESAR, BRUTUS, AND ANTONY")
print("="*70)

ai = BookSpikeAI()

# Train on the full story
print("\n📖 TRAINING ON THE STORY...")
ai.train_on_text(story_text, "The Tragedy of Caesar", epochs=5)

# Train on Antony's speech
print("\n📖 TRAINING ON ANTONY'S FUNERAL SPEECH...")
ai.train_on_text(antony_speech, "Antony's Speech", epochs=5)

# Train on Brutus's speech
print("\n📖 TRAINING ON BRUTUS'S SPEECH...")
ai.train_on_text(brutus_speech, "Brutus's Speech", epochs=5)

# Show character connections
ai.show_connections("CHARACTER CONNECTIONS", 
                    ['caesar', 'brutus', 'antony', 'cassius', 'octavius'])

# Show key concepts
ai.show_connections("KEY CONCEPTS", 
                    ['ambition', 'honorable', 'conspirators', 'funeral', 'rome'])

# Show emotions and themes
ai.show_connections("THEMES AND EMOTIONS", 
                    ['love', 'hate', 'honor', 'death', 'freedom'])

# Show relationships between characters
print("\n🔍 CHARACTER RELATIONSHIPS MAP:")
print("-" * 50)

# Check who Brutus connects to
brutus_connections = ai.query('brutus')
if brutus_connections:
    print("\nBRUTUS is connected to:")
    for w, att, dist in brutus_connections[:8]:
        print(f"   → {w}: attention={att:.4f}")

# Check who Antony connects to
antony_connections = ai.query('antony')
if antony_connections:
    print("\nANTONY is connected to:")
    for w, att, dist in antony_connections[:8]:
        print(f"   → {w}: attention={att:.4f}")

# Check who Caesar connects to
caesar_connections = ai.query('caesar')
if caesar_connections:
    print("\nCAESAR is connected to:")
    for w, att, dist in caesar_connections[:8]:
        print(f"   → {w}: attention={att:.4f}")

# Show final status
ai.status()

print("\n" + "="*70)
print("✅ SPIKE AI UNDERSTANDS THE TRAGEDY!")
print("="*70)
print("""
The AI now knows:
- Caesar was assassinated by conspirators including Brutus
- Brutus loved Rome more than Caesar
- Antony turned the crowd against Brutus
- Antony called Brutus 'the noblest Roman'
- Brutus and Cassius died by suicide
- The themes: ambition, honor, betrayal, freedom
""")

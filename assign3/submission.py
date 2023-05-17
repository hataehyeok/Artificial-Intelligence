import shell
import util
import wordsegUtil



############################################################
# Problem 1: Word Segmentation

# Problem 1a: Solve the word segmentation problem under a unigram model

class WordSegmentationProblem(util.SearchProblem):
    def __init__(self, query, unigramCost):
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return 0
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state == len(self.query)
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 5 lines of code, but don't worry if you deviate from this)
        possible_words = []
        for i in range(state, len(self.query)):
            cost = self.unigramCost(self.query[state:i+1])
            possible_words.append((self.query[state:i+1], i+1, cost))
        return possible_words
        # END_YOUR_ANSWER

def segmentWords(query, unigramCost):
    if len(query) == 0:
        return ''

    ucs = util.UniformCostSearch()
    ucs.solve(WordSegmentationProblem(query, unigramCost))

    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return ' '.join(ucs.actions)
    # END_YOUR_ANSWER

# Problem 1b: Solve the k-word segmentation problem under a unigram model

class KWordSegmentationProblem(util.SearchProblem):
    def __init__(self, k, query, unigramCost):
        self.k = k
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return 0, self.k
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        pos, remaining_words = state
        return pos == len(self.query) and remaining_words == 0
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 7 lines of code, but don't worry if you deviate from this)
        pos, remaining_words = state
        if remaining_words > 0:
            for end in range(pos + 1, len(self.query) + 1):
                word = self.query[pos:end]
                nextState = end, remaining_words - 1
                cost = self.unigramCost(word)
                yield (word, nextState, cost)
        # END_YOUR_ANSWER

def segmentKWords(k, query, unigramCost):
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch()
    ucs.solve(KWordSegmentationProblem(k, query, unigramCost))
    return ' '.join(ucs.actions)
    # END_YOUR_ANSWER

############################################################
# Problem 2: Vowel Insertion

# Problem 2a: Solve the vowel insertion problem under a bigram cost

class VowelInsertionProblem(util.SearchProblem):
    def __init__(self, queryWords, bigramCost, possibleFills):
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return 0, wordsegUtil.SENTENCE_BEGIN
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        pos, _ = state
        return pos == len(self.queryWords)
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 9 lines of code, but don't worry if you deviate from this)
        pos, prev_word = state
        possible_words = []
        cand_words = self.possibleFills(self.queryWords[pos])

        if len(cand_words) == 0:
            cand_words = {self.queryWords[pos]}

        for word in cand_words:
            new_state = (pos + 1, word)
            cost = self.bigramCost(prev_word, word)
            possible_words.append((word, new_state, cost))

        return possible_words
        # END_YOUR_ANSWER

def insertVowels(queryWords, bigramCost, possibleFills):
    # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch()
    ucs.solve(VowelInsertionProblem(queryWords, bigramCost, possibleFills))
    return ' '.join(ucs.actions)
    # END_YOUR_ANSWER

# Problem 2b: Solve the limited vowel insertion problem under a bigram cost

class LimitedVowelInsertionProblem(util.SearchProblem):
    def __init__(self, impossibleVowels, queryWords, bigramCost, possibleFills):
        self.impossibleVowels = impossibleVowels
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return 0, wordsegUtil.SENTENCE_BEGIN
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        pos, _ = state
        return pos == len(self.queryWords)
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 10 lines of code, but don't worry if you deviate from this)
        pos, prev_word = state
        possible_words = []
        cand_words = self.possibleFills(self.queryWords[pos])

        if len(cand_words) == 0:
            cand_words = {self.queryWords[pos]}

        for word in cand_words:
            if all(vowel not in word for vowel in self.impossibleVowels):
                new_state = (pos + 1, word)
                cost = self.bigramCost(prev_word, word)
                possible_words.append((word, new_state, cost))

        return possible_words
        # END_YOUR_ANSWER

def insertLimitedVowels(impossibleVowels, queryWords, bigramCost, possibleFills):
    # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch()
    ucs.solve(LimitedVowelInsertionProblem(impossibleVowels, queryWords, bigramCost, possibleFills))

    if ucs.actions is None:
        return ' '.join(w.replace(' ', '') for w in queryWords)
    else:
        return ' '.join(ucs.actions)
    # END_YOUR_ANSWER

############################################################
# Problem 3: Putting It Together

# Problem 3a: Solve the joint segmentation-and-insertion problem

class JointSegmentationInsertionProblem(util.SearchProblem):
    def __init__(self, query, bigramCost, possibleFills):
        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return 0, wordsegUtil.SENTENCE_BEGIN
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        pos, _ = state
        return pos == len(self.query)
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)
        pos, prev_word = state
        possible_words = []

        for i in range(pos, len(self.query)):
            for word in self.possibleFills(self.query[pos:i + 1]):
                new_state = (i + 1, word)
                cost = self.bigramCost(prev_word, word)
                possible_words.append((word, new_state, cost))
        return possible_words
        # END_YOUR_ANSWER

def segmentAndInsert(query, bigramCost, possibleFills):
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch()
    ucs.solve(JointSegmentationInsertionProblem(query, bigramCost, possibleFills))
    return ' '.join(ucs.actions)
    # END_YOUR_ANSWER

############################################################
# Problem 4: A* search

# Problem 4a: Define an admissible but not consistent heuristic function

class SimpleProblem(util.SearchProblem):
    def __init__(self):
        # BEGIN_YOUR_ANSWER (our solution is 4 lines of code, but don't worry if you deviate from this)
        self.start_state = "A"
        self.state_B = "B"
        self.state_C = "C"
        self.end_state = "D"
        # END_YOUR_ANSWER

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return self.start_state
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state == self.end_state
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
        if state == self.start_state:
            return [("A->B", self.state_B, 3), ("A->C", self.state_C, 1)]
        elif state == self.state_B:
            return [("B->D", self.end_state, 1)]
        elif state == self.state_C:
            return [("C->B", self.state_B, 1)]
        # END_YOUR_ANSWER

def admissibleButInconsistentHeuristic(state):
    # BEGIN_YOUR_ANSWER (our solution is 2 lines of code, but don't worry if you deviate from this)
    h = {"A": 3, "B": 0, "C": 2, "D": 0}
    return h[state]
    # END_YOUR_ANSWER

# Problem 4b: Apply a heuristic function to the joint segmentation-and-insertion problem

def makeWordCost(bigramCost, wordPairs):
    """
    :param bigramCost: learned bigram cost from a training corpus
    :param wordPairs: all word pairs in the training corpus
    :returns: wordCost, which is a function from word to cost
    """
    # BEGIN_YOUR_ANSWER (our solution is 4 lines of code, but don't worry if you deviate from this)
    costs_word = {}
    for word1, word2 in wordPairs:
        cost = bigramCost(word1, word2)
        if word2 not in costs_word or cost < costs_word[word2]:
            costs_word[word2] = cost
    
    def word_cost(w):
        return costs_word.get(w, bigramCost(wordsegUtil.SENTENCE_UNK, w))
    
    return word_cost
    # END_YOUR_ANSWER

class RelaxedProblem(util.SearchProblem):
    def __init__(self, query, wordCost, possibleFills):
        self.query = query
        self.wordCost = wordCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return 0
        # END_YOUR_ANSWER

    def isEnd(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state == len(self.query)
        # END_YOUR_ANSWER

    def succAndCost(self, state):
        # BEGIN_YOUR_ANSWER (our solution is 5 lines of code, but don't worry if you deviate from this)
        cand_words = []
        for _ in range(1, len(self.query) - state + 1):
            next_state = state + _
            word = self.query[state:next_state]
            possible_words = self.possibleFills(word)
            for possible_word in possible_words:
                cand_words.append((possible_word, next_state, self.wordCost(possible_word)))
        return cand_words
        # END_YOUR_ANSWER

def makeHeuristic(query, wordCost, possibleFills):
    # BEGIN_YOUR_ANSWER (our solution is 2 lines of code, but don't worry if you deviate from this)
    relaxed_problem = RelaxedProblem(query, wordCost, possibleFills)
    dp = util.DynamicProgramming(relaxed_problem)
    
    def heuristic(state):
        return dp(state[0])
    
    return heuristic
    # END_YOUR_ANSWER

def fastSegmentAndInsert(query, bigramCost, wordCost, possibleFills):
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_ANSWER (our solution is 4 lines of code, but don't worry if you deviate from this)
    heuristic = makeHeuristic(query, wordCost, possibleFills)
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(JointSegmentationInsertionProblem(query, bigramCost, possibleFills), heuristic)
    return ' '.join(ucs.actions)
    # END_YOUR_ANSWER

############################################################

if __name__ == '__main__':
    shell.main()

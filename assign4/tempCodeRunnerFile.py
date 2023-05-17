for i, count in enumerate(counts):
                    if count > 0:
                        newTotal = total + self.cardValues[i]
                        newCounts = counts[:i] + (count - 1,) + counts[i+1:] if sumCounts > 1 else None
                        reward = newTotal if newCounts is None else 0
                        results.append(((newTotal, None, newCounts if newTotal <= self.threshold else None), count / sumCounts, reward if newTotal <= self.threshold else 0))
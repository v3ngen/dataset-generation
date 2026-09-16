# Stakeholders and Cost Matrices

**Audience: students.** The prose in §1-§3 is written to be lifted into the assignment brief.
§4-§6 are the mechanics: the matrices, how to use them, and what is expected. §7 is a note for
the unit leader on how the numbers were chosen.

Both cost matrices are provided in code in `costs.py` and are already imported and used in the
starter notebook, alongside accuracy. You do not need to type them in.

---

## 1. The scenario

A pan-European research consortium ran a lifestyle-and-health survey across Italy, France, the
United Kingdom and Norway, recording each respondent's demographics, coffee and caffeine
consumption, sleep, stress, physical activity, smoking and drinking behaviour, and a set of
routine physical measurements. Each respondent was then followed for the next twelve months, and
the consortium recorded whether they went on to have a **year of high health needs**: fourteen or
more days of health-related absence from work or normal activity, or six or more primary-care
contacts.

That follow-up flag, `HighHealthNeeds`, is what you are asked to predict.

Two organisations want to use your model. Both run the same intervention — a twelve-week
preventive lifestyle programme, offered by invitation — and both will run your model over the
same people to decide who to invite. They are not, however, the same customer, and they will not
agree about which of your models is best.

## 2. Stakeholder A — a national public health agency

The agency runs the programme as a free, publicly funded service: a health check followed by a
twelve-week lifestyle course, offered by invitation. Its mandate is population health and its
budget is judged against **downstream treatment costs**, so the thing that worries it is the
person it fails to invite. Someone who goes on to have a high-needs year without ever being
offered support represents avoidable primary and secondary care spending and, in the agency's own
framing, an equity failure — the people most likely to be missed are precisely the people least
likely to present voluntarily. Programme places are inexpensive and the agency would far rather
over-invite than under-invite: an unnecessary invitation costs it a health check and a course
place, and the person invited comes to no harm. Its question about any model you hand it is
blunt: **how many of the people who needed help did we miss, and what did that cost us?**

## 3. Stakeholder B — an employer's occupational health team

The employer buys a small, fixed block of places on the same programme each year for its
workforce, signed off by finance with no capacity to overspend. Every place given to someone who
was not going to have a high-needs year is a place a colleague who needed it did not get, and
finance sees it as money spent for no measurable return. The employer does carry a real cost when
it misses someone — absence cover, lost productivity, temporary staffing — but that cost is a
fraction of what the health agency absorbs, and it is spread across the business rather than
landing on the occupational health budget. Its question is close to the opposite of the agency's:
**of the people you told us to invite, how many actually needed it, and can we defend the
spend?**

---

## 4. The cost matrices

Costs are per person scored, in euros. Negative numbers are **net benefits** — cases where the
intervention pays for itself and more.

Rows and columns both lead with the positive case (high-needs / invite), so correct calls sit on
the diagonal running top-left to bottom-right — **TP** top-left, **TN** bottom-right — matching
the usual confusion-matrix convention.

### Stakeholder A — national public health agency

|  | Predicted: invite | Predicted: no invitation |
|---|---|---|
| **Actually high-needs** | **EUR -720** (TP) — programme cost of 180 offset by an expected 900 in avoided care | **EUR 1,450** (FN) — avoidable downstream primary and secondary care |
| **Actually not high-needs** | **EUR 180** (FP) — wasted health check and programme place | EUR 0 (TN) |

### Stakeholder B — employer occupational health

|  | Predicted: invite | Predicted: no invitation |
|---|---|---|
| **Actually high-needs** | **EUR -80** (TP) — programme cost of 520 offset by an expected 600 in avoided absence | **EUR 780** (FN) — absence cover, temporary staffing, lost productivity |
| **Actually not high-needs** | **EUR 520** (FP) — a scarce programme place consumed for no return | EUR 0 (TN) |

Note the asymmetry. For the agency a false negative costs about **eight times** a false positive.
For the employer the ratio is closer to **1.5 to 1**, and in the opposite direction from what the
agency would do about it.

---

## 5. Using the matrices: from predictions to money

### 5.1 Total cost and cost per person

Score a set of predictions against `y_true` and you get a confusion matrix: TN, FP, FN, TP —
counts of people, not money. Multiply each count by what that cell costs in the stakeholder's
matrix and add them up:

```
total_cost = TN*C_TN + FP*C_FP + FN*C_FN + TP*C_TP
```

**Cost per person** is that same total divided by how many people you scored:

```
cost_per_person = total_cost / n              where n = TN + FP + FN + TP
```

Use cost per person whenever you compare across different-sized groups — the ~2,000-row
validation split against the smaller held-out test set, say — since the raw total scales with
group size and isn't otherwise comparable.

**Worked example.** Ten people, scored with a threshold of 0.5 (defined properly in §5.2), giving
a confusion matrix of TN=5, FP=0, FN=3, TP=2 (n=10). Under the public health agency's matrix
(TN=€0, FP=€180, FN=€1,450, TP=−€720):

```
total_cost      = 5×0 + 0×180 + 3×1,450 + 2×(−720)
                = 0 + 0 + 4,350 − 1,440
                = €2,910

cost_per_person = 2,910 / 10 = €291.00
```

`costs.py` does both steps for you — `total_cost(y_true, y_pred, matrix)` and
`cost_per_person(y_true, y_pred, matrix)` — so in practice you call the function, not the formula.
The arithmetic above is here so you know what the function is doing.

### 5.2 From a probability to a decision

A classifier doesn't hand you "invite" or "don't invite" directly. `model.predict_proba(X)[:, 1]`
gives you, for each person, the model's estimate of the probability they will go on to have a
year of high health needs — a number between 0 and 1, not a decision.

Turning that probability into a decision needs a **threshold** `t`: predict positive (invite) if
the probability is at least `t`, otherwise predict negative. `model.predict(X)` does exactly this
with `t` fixed at 0.5, silently — call `.predict()` and you have already made a threshold
decision, whether you meant to or not.

```python
probabilities = model.predict_proba(X_val)[:, 1]     # P(high health needs), one per person

predictions = (probabilities >= 0.5).astype(int)      # what model.predict() does for you
```

There is nothing about the number 0.5 that connects to either stakeholder's costs. It is what you
get if you never think about the threshold at all.

### 5.3 The threshold that minimises expected cost

For one person, with the model's estimated probability `p` that they are a positive case, the
**expected** cost of each action follows directly from the matrix:

```
E[cost | invite]     = p × C_TP + (1 − p) × C_FP
E[cost | don't invite] = p × C_FN + (1 − p) × C_TN
```

Inviting is the cheaper action exactly when `E[cost | invite] < E[cost | don't invite]`. Solve
that inequality for `p` and the two sides become equal at:

```
p* = (C_FP − C_TN) / ((C_FP − C_TN) + (C_FN − C_TP))
```

**This is computed once, from the four cost numbers alone.** It does not depend on your model,
your data, or any confusion matrix — it is a property of the stakeholder's economics, not of your
predictions. `costs.py`'s `optimal_threshold(matrix)` computes it directly from `COST_PUBLIC_HEALTH`
or `COST_EMPLOYER`.

For the public health agency:

```
C_FP − C_TN = 180 − 0     = 180        (the extra cost of an unnecessary invitation)
C_FN − C_TP = 1,450 − (−720) = 2,170   (the extra cost of a missed case)

p* = 180 / (180 + 2,170) = 180 / 2,350 = 0.077
```

For the employer:

```
C_FP − C_TN = 520 − 0     = 520
C_FN − C_TP = 780 − (−80) = 860

p* = 520 / (520 + 860) = 520 / 1,380 = 0.377
```

Read the formula as a share: `p*` is the fraction of total mistake-cost that comes from false
alarms. When a miss is far more expensive than a false alarm (the agency: €2,170 against €180),
that share is small, so `p*` sits close to 0 — invite almost anyone with a meaningful chance of
being a positive case, because being wrong the false-alarm way is cheap. When the two mistakes
are closer in cost (the employer: €860 against €520), `p*` sits nearer the middle — invite only
people you are reasonably confident about.

**Checking it against two people.** Take someone the model rates at `p = 0.15` — above the
agency's 0.077 threshold:

```
E[cost | invite]       = 0.15×(−720) + 0.85×180 = −108 + 153 = €45
E[cost | don't invite] = 0.15×1,450  + 0.85×0    = €217.50

€45 < €217.50  ->  inviting is cheaper. Invite.
```

And someone rated `p = 0.05` — below it:

```
E[cost | invite]       = 0.05×(−720) + 0.95×180 = −36 + 171 = €135
E[cost | don't invite] = 0.05×1,450  + 0.95×0    = €72.50

€135 > €72.50  ->  not inviting is cheaper. Don't invite.
```

The same check for the employer, at `p = 0.40` (above their 0.377) and `p = 0.25` (below it):

```
p = 0.40:  E[invite] = 0.40×(−80) + 0.60×520 = €280   E[don't invite] = 0.40×780 = €312   -> invite
p = 0.25:  E[invite] = 0.25×(−80) + 0.75×520 = €370   E[don't invite] = 0.25×780 = €195   -> don't invite
```

Quick reference:

| Stakeholder | `p*` | Behaviour |
|---|---|---|
| A — public health agency | **0.077** | Invite aggressively; tolerate many false positives to avoid misses |
| B — employer | **0.377** | Invite only where reasonably confident; protect the scarce places |

**The default 0.5 is wrong for both of them.** Neither stakeholder's interests are served by
optimising accuracy — §5.4 shows exactly how wrong, in money.

### 5.4 Worked example: the same ten people, three thresholds

Ten people from a validation split, with the model's estimated probability and whether they
actually went on to have a year of high health needs. *(Ten is small enough to follow by hand —
real work uses your full validation split, which is a few thousand rows. The 50/50 split of
outcomes below is exaggerated for the same reason; your real data is roughly 80/20.)*

| Person | Actually high-needs? | Model's `p` | Invite @ 0.50? | Invite @ 0.077 (agency)? | Invite @ 0.377 (employer)? |
|---|---|---|---|---|---|
| 1 | No | 0.02 | No | No | No |
| 2 | No | 0.05 | No | No | No |
| 3 | No | 0.08 | No | **Yes** | No |
| 4 | **Yes** | 0.10 | No | **Yes** | No |
| 5 | No | 0.15 | No | **Yes** | No |
| 6 | No | 0.25 | No | **Yes** | No |
| 7 | **Yes** | 0.30 | No | **Yes** | No |
| 8 | **Yes** | 0.40 | No | **Yes** | **Yes** |
| 9 | **Yes** | 0.55 | **Yes** | **Yes** | **Yes** |
| 10 | **Yes** | 0.85 | **Yes** | **Yes** | **Yes** |

Same ten probabilities, three different sets of decisions — nothing about the model changed
between columns, only the threshold applied to its output. Reading off the confusion matrix at
each threshold and pricing it under both matrices:

| Threshold | Confusion (TP, FN, FP, TN) | Cost/person — agency (A) | Cost/person — employer (B) |
|---|---|---|---|
| 0.50 (default) | 2, 3, 0, 5 | €291.00 | €218.00 |
| 0.077 (agency's own) | 5, 0, 3, 2 | **−€306.00** | €116.00 |
| 0.377 (employer's own) | 3, 2, 0, 5 | €74.00 | **€132.00** |

At the default threshold the agency pays €291 per person. At its own threshold it makes a
**€306 net saving per person** instead — a swing of nearly €600 per person from a decision that
costs nothing to make, since it uses probabilities the model already produces. The employer's own
threshold does the same job at a smaller scale, cutting their cost from €218 to €132.

Every number in both tables is exactly the confusion-matrix arithmetic of §5.1, run three times
against the same ten probabilities. Nothing here is estimated or approximate — you can reproduce
every cell by hand from the table above, and `costs.py` reproduces it for your actual validation
predictions in three function calls:

```python
from costs import COST_PUBLIC_HEALTH, COST_EMPLOYER, cost_per_person, optimal_threshold

probabilities = model.predict_proba(X_val)[:, 1]

for name, matrix in [('agency', COST_PUBLIC_HEALTH), ('employer', COST_EMPLOYER)]:
    t = optimal_threshold(matrix)
    predictions = (probabilities >= t).astype(int)
    print(name, t, cost_per_person(y_val, predictions, matrix))
```

---

## 6. What you are expected to do with this

1. Report **both** stakeholders' total and per-person cost for every model you evaluate, not just
   accuracy.
2. Treat the decision threshold as something you tune per stakeholder, and show the effect.
3. Expect the two stakeholders to **disagree about which model is best**, and when they do, say
   so explicitly and explain *why* in terms of the trade-off between precision and recall. A
   model that is better for one may be a poor choice for the other. Identifying that disagreement
   and quantifying it is worth more marks than squeezing out another point of accuracy.
4. Make a recommendation to each stakeholder, and justify it in their terms — not in yours.

A high-accuracy model that misses most of the high-needs cases is a bad model for the agency and
you should be able to prove it with numbers rather than assert it.

---

## 7. Note for the unit leader

The figures are plausible rather than sourced, and are chosen to produce specific pedagogical
behaviour:

- The two cost-optimal thresholds (0.077 and 0.377) sit either side of 0.5 in usefulness terms and
  are far apart, so threshold tuning is worth a great deal and a single operating point cannot
  serve both stakeholders.
- The ratios are asymmetric in *opposite directions*, which is what makes a genuine rank reversal
  between a recall-oriented and a precision-oriented model possible.
- Both matrices give a net benefit for true positives, so students see that a correct positive
  prediction creates value rather than merely avoiding loss — which is what makes the agency's
  very low optimal threshold intuitive rather than arbitrary.

`validate_ml_ladder.py` asserts that a rank reversal actually occurs between two plausible
student models under these numbers. If the figures are changed, re-run it.

Currency is euros throughout for simplicity, even though the four countries in the dataset do not
share one. If that is distracting, it can be relabelled as "cost units" without changing anything.

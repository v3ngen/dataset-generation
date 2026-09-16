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

## 5. Using the matrices

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

Use cost per person whenever you compare across different-sized groups — your validation split
against the smaller held-out test set, say — since the raw total scales with group size and isn't
otherwise comparable.

Nothing here needs anything beyond `model.predict()`. You produce predictions exactly as you
already do, build the confusion matrix exactly as you already do, and price it.

**Worked example.** Ten people, of whom five actually went on to have a year of high health needs.
The model correctly identified two of them, missed three, and wrongly invited nobody — a confusion
matrix of TP=2, FN=3, FP=0, TN=5 (n=10). Under the public health agency's matrix (TP=−€720,
FN=€1,450, FP=€180, TN=€0):

```
total_cost      = 2×(−720) + 3×1,450 + 0×180 + 5×0
                = −1,440 + 4,350 + 0 + 0
                = €2,910

cost_per_person = 2,910 / 10 = €291.00
```

`costs.py` does both steps for you — `total_cost(y_true, y_pred, matrix)` and
`cost_per_person(y_true, y_pred, matrix)`, plus `cost_summary(y_true, y_pred)` for both
stakeholders at once. The arithmetic above is here so you know what the function is doing; run
`python3 costs.py` to see this exact example computed.

### 5.2 Worked example: why the stakeholders disagree

Here is the whole point of having two matrices. Two models, scored on the same 100 people, of whom
20 actually had a year of high health needs.

**Model R** was trained with `class_weight='balanced'`. It casts a wide net: it catches most of the
people who needed help, at the price of a lot of unnecessary invitations.

**Model P** was trained without class weighting. It is cautious: when it says "invite" it is
usually right, but it misses more than half the people who needed help.

| | Model R (class-weighted) | Model P (unweighted) |
|---|---|---|
| TP — correctly invited | 15 | 8 |
| FN — missed | 5 | 12 |
| FP — wrongly invited | 25 | 4 |
| TN — correctly not invited | 55 | 76 |
| **Accuracy** | 0.70 | **0.84** |
| **Precision** | 0.375 | **0.667** |
| **Recall** | **0.75** | 0.40 |
| **F1** | 0.500 | 0.500 |

On the traditional metrics, Model P looks like the better model: it is 14 points more accurate and
nearly twice as precise. And their F1 scores are **identical** — the metrics cannot separate them.

Now price both confusion matrices under each stakeholder's costs:

```
Model R, agency:    15×(−720) + 5×1,450  + 25×180 + 55×0 = €950      -> €9.50   per person
Model P, agency:     8×(−720) + 12×1,450 +  4×180 + 76×0 = €12,360   -> €123.60 per person

Model R, employer:  15×(−80)  + 5×780    + 25×520 + 55×0 = €15,700   -> €157.00 per person
Model P, employer:   8×(−80)  + 12×780   +  4×520 + 76×0 = €10,800   -> €108.00 per person
```

| | Model R | Model P | Who wins |
|---|---|---|---|
| Accuracy | 0.70 | **0.84** | Model P |
| F1 | 0.500 | 0.500 | a tie |
| **Cost to the agency** | **€9.50** | €123.60 | **Model R** |
| **Cost to the employer** | €157.00 | **€108.00** | **Model P** |

**Four criteria, three different answers.** Accuracy prefers Model P. F1 cannot tell them apart.
The agency would pay thirteen times more per person for Model P. The employer would pay about 45%
more per person for Model R.

Neither stakeholder is wrong, and neither model is wrong. The agency's missed cases cost it €1,450
each, so a model that misses twelve people out of a hundred is ruinous to it — the 25 unnecessary
invitations Model R makes are cheap by comparison. The employer pays €520 for each of those
unnecessary invitations out of a fixed budget, and only €780 when it misses someone, so the same
wide net is poor value.

This is why "which model is best?" cannot be answered from the metrics alone, and why you are
asked to report cost per person for both stakeholders alongside them.

---

## 6. What you are expected to do with this

1. Report **both** stakeholders' total and per-person cost for every model you evaluate, not just
   accuracy.
2. Show how the choices you make in the pipeline — class weighting or resampling above all —
   move the cost for each stakeholder, not just the accuracy.
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

- The ratios are asymmetric in *opposite directions* — roughly 8:1 for the agency, 1.5:1 for the
  employer — which is what makes a genuine rank reversal between a recall-oriented and a
  precision-oriented model possible.
- Both matrices give a net benefit for true positives, so students see that a correct positive
  prediction creates value rather than merely avoiding loss, rather than every cell being a
  penalty.
- The magnitudes are set so that class weighting alone flips the agency from paying out to making
  a net saving. That keeps the whole exercise reachable with `model.predict()` and the standard
  metrics: no predicted probabilities and no decision thresholds are needed anywhere.

`validate_ml_ladder.py` rung 8 asserts that at least one pair of plausible student models splits
the two stakeholders, and that the most accurate model is what neither of them wants.

If the figures are changed, re-run `validate_ml_ladder.py`.

Currency is euros throughout for simplicity, even though the four countries in the dataset do not
share one. If that is distracting, it can be relabelled as "cost units" without changing anything.

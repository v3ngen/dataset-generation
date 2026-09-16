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

Every number is what the stakeholder **spends** on that person over the following year, in euros.
No cell is negative: a correctly invited person still costs the programme fee, plus whatever
burden the programme does not manage to avert. Nobody makes money — the goal is to spend less.

Both organisations assume the twelve-week programme averts about **55%** of the burden for
someone it reaches. That is where the "correctly invited" figures come from.

Rows and columns both lead with the positive case (high-needs / invite), so correct calls sit on
the diagonal running top-left to bottom-right — **TP** top-left, **TN** bottom-right — matching
the usual confusion-matrix convention.

### Stakeholder A — national public health agency

|  | Predicted: invite | Predicted: no invitation |
|---|---|---|
| **Actually high-needs** | **EUR 870** (TP) — EUR 150 for the place, plus the EUR 720 of care the programme cannot avert | **EUR 1,600** (FN) — a full year of avoidable primary and secondary care |
| **Actually not high-needs** | **EUR 150** (FP) — health check and programme place, wasted | EUR 0 (TN) — nothing happens, nothing is spent |

### Stakeholder B — employer occupational health

|  | Predicted: invite | Predicted: no invitation |
|---|---|---|
| **Actually high-needs** | **EUR 1,260** (TP) — EUR 450 for the place, plus the EUR 810 of absence still not averted | **EUR 1,800** (FN) — absence cover, temporary staffing, lost productivity |
| **Actually not high-needs** | **EUR 450** (FP) — a scarce block-booked place consumed for no return | EUR 0 (TN) — nothing happens, nothing is spent |

### What the asymmetry actually is

Comparing two models only ever depends on two quantities: what it costs you to invite someone
who did not need it, and what it costs you to miss someone who did.

| | Cost of a **false alarm**<br>(FP − TN) | Cost of a **miss**<br>(FN − TP) | Ratio |
|---|---|---|---|
| **Agency** | EUR 150 | EUR 730 | **4.9 : 1** — misses hurt far more, so cast a wide net |
| **Employer** | EUR 450 | EUR 540 | **1.2 : 1** — roughly balanced, so be selective |

The agency's places are cheap and the care it avoids is expensive, so it would much rather
over-invite. The employer's places are scarce and expensive while the absence it avoids is only
partly recoverable, so a wasted place costs it nearly as much as a miss. Those two ratios are why
the same model can be a good buy for one and a bad buy for the other.

---

## 5. Using the matrices

### 5.1 Total cost and cost per person

Score a set of predictions against `y_true` and you get a confusion matrix: TP, FN, FP, TN —
counts of people, not money. Multiply each count by what that cell costs and add them up:

```
total_cost = TP*C_TP + FN*C_FN + FP*C_FP + TN*C_TN
```

**Cost per person** is that same total divided by how many people you scored:

```
cost_per_person = total_cost / n              where n = TP + FN + FP + TN
```

Use cost per person whenever you compare across different-sized groups — your validation split
against the smaller held-out test set, say — since the raw total scales with group size.

Nothing here needs anything beyond `model.predict()`.

**Worked example.** Ten people, of whom five went on to have a year of high health needs. The
model correctly identified two, missed three, and wrongly invited nobody — TP=2, FN=3, FP=0,
TN=5:

```
agency total = 2×870 + 3×1,600 + 0×150 + 5×0
             = 1,740 + 4,800
             = EUR 6,540          ->  6,540 / 10 = EUR 654.00 per person
```

### 5.2 The number that gives cost its meaning: doing nothing

A cost per person means very little on its own. The reference point is **what it would cost to
invite nobody at all** — no programme, no model, just pay for everyone you missed.

For the ten people above, all five high-needs cases would be missed:

```
agency, invite nobody = 5 × 1,600 = EUR 8,000   ->  EUR 800.00 per person
```

So that model is worth having: €654 against €800. `costs.py` gives you this as
`do_nothing_cost(y_true, matrix)`. **A model that cannot beat it is not worth deploying**, however
good its accuracy looks, and you should report both numbers together.

### 5.3 Worked example: why the stakeholders disagree

Two models, scored on the same 100 people, of whom 20 actually had a year of high health needs.

**Model R** was trained with `class_weight='balanced'` — a wide net, catching most of the people
who needed help at the price of many unnecessary invitations. **Model P** was trained without it —
cautious, usually right when it says "invite", but missing more than half the cases.

The third column is the do-nothing baseline from §5.2 — invite nobody — scored the same way, so
it can be compared against directly.

| | Model R (class-weighted) | Model P (unweighted) | Invite nobody |
|---|---|---|---|
| TP — correctly invited | 15 | 8 | 0 |
| FN — missed | 5 | 12 | 20 |
| FP — wrongly invited | 25 | 4 | 0 |
| TN — correctly not invited | 55 | 76 | 80 |
| **Accuracy** | 0.70 | **0.84** | 0.80 |
| **Precision** | 0.375 | **0.667** | *undefined* |
| **Recall** | **0.75** | 0.40 | 0.00 |
| **F1** | 0.500 | 0.500 | *undefined* |

On the traditional metrics Model P looks better: 14 points more accurate and nearly twice as
precise. Their F1 scores are **identical** — the metrics cannot separate them at all.

**Why two of those cells say "undefined".** Precision is `TP / (TP + FP)` — of the people you
invited, how many needed it. Invite nobody and that is `0 / 0`: there is no set of invitations to
be right or wrong about, so the question has no answer. F1 is built from precision, so it is
undefined too.

Worth knowing because you will meet this in code: `precision_score` and `f1_score` both **return
0.0** here rather than failing, and raise an `UndefinedMetricWarning` saying the metric is
ill-defined. A 0.0 that actually means "not applicable" is easy to read as "scored zero", and the
two are not the same thing. Recall and accuracy are both perfectly well defined — 0.00 and 0.80 —
and that pairing is the whole problem with accuracy on this dataset in one line.

Now price both, against the do-nothing reference:

```
Model R, agency:    15×870 + 5×1,600  + 25×150 = EUR 24,800  ->  EUR 248.00 per person
Model P, agency:     8×870 + 12×1,600 +  4×150 = EUR 26,760  ->  EUR 267.60 per person
        agency, invite nobody:        20×1,600 = EUR 32,000  ->  EUR 320.00 per person

Model R, employer:  15×1,260 + 5×1,800  + 25×450 = EUR 39,150  ->  EUR 391.50 per person
Model P, employer:   8×1,260 + 12×1,800 +  4×450 = EUR 33,480  ->  EUR 334.80 per person
        employer, invite nobody:         20×1,800 = EUR 36,000  ->  EUR 360.00 per person
```

| | Model R | Model P | Invite nobody | Who wins |
|---|---|---|---|---|
| Accuracy | 0.70 | **0.84** | 0.80 | Model P |
| Precision | 0.375 | **0.667** | *undefined* | Model P |
| Recall | **0.75** | 0.40 | 0.00 | Model R |
| F1 | 0.500 | 0.500 | *undefined* | a tie |
| **Cost to the agency** | **€248.00** | €267.60 | €320.00 | **Model R** |
| **Cost to the employer** | €391.50 | **€334.80** | €360.00 | **Model P** |

**Four criteria, four different answers.** Accuracy and precision prefer Model P. Recall prefers
Model R. F1 cannot tell them apart. The agency prefers Model R; the employer prefers Model P.

Note where the do-nothing column sits: it beats Model R on accuracy, and it is only four points
behind Model P — a "model" that does nothing at all, requires no data and finds nobody is within
four points of your best accuracy score. It is last on recall and last on cost for the agency,
which is where its uselessness finally shows up.

And there is a fourth finding hiding in the table, which is the one worth writing up:
**Model R is worse than useless to the employer.** At €391.50 per person it costs more than not
running the programme at all (€360.00). A model can be a perfectly good model, clearly better than
nothing for one customer, and still be something the other customer should refuse to deploy.

Neither stakeholder is wrong. The agency pays €1,600 for every miss and only €150 for a wasted
place, so Model R's 25 unnecessary invitations are cheap next to the seven extra cases it catches.
The employer pays €450 a place out of a fixed budget and recovers only part of the absence cost,
so the same wide net destroys value.

This is why "which model is best?" cannot be answered from the metrics alone, and why you are
asked to report cost per person for both stakeholders — against the do-nothing baseline —
alongside them.

---

## 6. What you are expected to do with this

1. Report **both** stakeholders' cost per person for every model you evaluate, not just accuracy,
   and always alongside the do-nothing baseline so the number means something.
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

- The false-alarm to miss ratios are roughly 4.9:1 for the agency and 1.2:1 for the employer,
  which is what makes a genuine rank reversal between a recall-oriented and a precision-oriented
  model possible.
- Every cell is a real cost, so a total is never a profit. An earlier version credited a correct
  invitation with the care it avoided while still charging the full care cost for a miss, which
  double-counted two different baselines and made the agency appear to make money. Each matrix now
  reads as "what we spend on this person", with `do_nothing_cost()` as the reference point.
- The magnitudes are set so the programme is clearly worth running for the agency (the best model
  saves it around EUR 90 per person against doing nothing) and only marginally so for the employer
  (around EUR 20), which is what makes the employer genuinely choosy.
- The magnitudes are set so that class weighting alone flips the agency from paying out to making
  a net saving. That keeps the whole exercise reachable with `model.predict()` and the standard
  metrics: no predicted probabilities and no decision thresholds are needed anywhere.

`validate_ml_ladder.py` rung 8 asserts that at least one pair of plausible student models splits
the two stakeholders, and that the most accurate model is what neither of them wants.

If the figures are changed, re-run `validate_ml_ladder.py`.

Currency is euros throughout for simplicity, even though the four countries in the dataset do not
share one. If that is distracting, it can be relabelled as "cost units" without changing anything.

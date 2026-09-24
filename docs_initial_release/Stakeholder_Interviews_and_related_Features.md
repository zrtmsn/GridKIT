# Stakeholder Interviews and Related Features

This file documents stakeholder interviews and their impact on development. It shows which questions were asked during interviews, what the most relevant answers were, and which of GridKIT's features directly relate to stakeholder feedback.

For GDPR compliance, interviewees are anonymized; the interviews are simply referred to by a number.


## Question pool

The questions below are grouped thematically only. The resulting order is **not representative** of the planned interview flow or the interview strategy. This document merely serves to collect possible questions to ask the stakeholders.

### Project focus

**Project focus:** Which focus is more interesting to you — the simulation of human behavior on the electricity market (demand with respect to dynamic electricity tariffs) or the implications for the distribution grid and its level of expansion?

**Dashboard outputs:** Which concrete outputs do you need on the dashboard? From your point of view, do further development decisions depend on them?

**Use cases:** In which types of decision-making processes could such a solution be helpful — setting tariff rates, investment planning, or others?

**Industry gaps:** Is there a gap you notice as an industry practitioner that we as students might be missing? *(Closing question)*

### Modelling and data

**Network granularity:** Which sizes of sub-networks are relevant for you (individual households, street segments, neighbourhoods)? How granular are the data you usually work with (rather general estimates or specific measurement data)?

**Further load sources:** Apart from electric vehicles — which further sources of high grid load would be interesting to model?

**Baselines:** How meaningful do baselines seem to you as reference points for comparisons?

**Configurability of the network model:** Which configuration options are necessary (e.g. degree of roll-out for wallboxes and heat pumps — per area or per household? Adjustment of transformer and line capacities? Configuration via map/UI and/or CSV)?

**Level of detail of the configuration:** What would you like to configure at which level of detail (high-level: share of households with a heat pump in the area; low-level: assigning a single electric vehicle to a specific household)?

**Data import:** Would you like to import your own network models? In which format are your network data available? Should your own load profiles, weather data, or similar also be usable?

### User-Interface (MapUI and Dashboard)

**Area selection:**

- Should it also be possible to select by transformer or by feeder / network segment?

- If households lie outside the selection window but are connected to a feeder whose transformer lies inside the selected area — how would you prefer these households to be handled? Should they be selected automatically, or should the system strictly stick to the selection made on the map?

**Must-have criteria:** What would be a must-have criterion in the user interface for a first usable version?

**Metrics and explainability:** Which metrics — in addition to the ones already available — would be useful to ensure that the analysis is accurate enough and therefore trustworthy? Which outputs or plots are needed? Is an export of the data, e.g. as CSV, desired?



### § 14a EnWG and grid congestion

*§ 14a of the German Energy Industry Act (EnWG — Energiewirtschaftsgesetz).*

**Grid congestion:** Which factors most favor grid congestion in your experience (e.g. loads, network topology)?

**Perception of § 14a EnWG:** How do you perceive § 14a EnWG? Do you consider applying it?

**Handling so far:** How have you dealt with § 14a EnWG so far? Are there ambitions or ongoing projects?

**Alternatives to § 14a EnWG:** What alternative measures for grid relief exist?

**Contacts with other stakeholders:** Do you cooperate or correspond with other companies (energy suppliers, grid expansion companies, etc.) or with the German Federal Ministry for Economic Affairs and Energy regarding grid congestion? How do these contacts perceive grid congestion?



### Computing-Infrastructure

**Infrastructure:** What do your computing clusters run on (AWS, Azure, Kubernetes, On-Premise)?



## Feedback from Interview 1

### Additional metrics

Show how much heat pumps and how much EVs contribute, with respect to their impact on curtailment. This should provide confidence that the modelling is sensible — verification that not only a single factor determines the entire curtailment.

### Low-level household configuration

The wallbox provides data on where e.g. an EV is charged (=> it is clear which household uses which relevant devices). It is unclear whether individual household configuration (e.g. adding an EV to a specific household) is actually applied in practice to simulate certain scenarios. It was nevertheless perceived as a good idea for extended functionality.

### Cluster infrastructure

All four options are used (Azure, AWS, Kubernetes, On-Premise), but nothing is based on SLURM as a batch system. The reason is that in the stakeholder's company, HPC resources do not need to be used by as many people simultaneously as at universities in research networks. Primarily AZURE/AWS are used at the stakeholder's company for software like ours.

### Suggestion for further interviews with other stakeholders

Ask how the interviewees understand our charts/plots: "What does this graph tell you?"



## Feedback from Interview 2

### Important input for further development decisions: Where is the biggest gap?

Interesting questions that frequently arise in daily work are the differences between urban areas and rural areas as well as seasonal variations (autumn, winter, spring, summer).

### What do we miss when looking from the outside?

- Dynamic pricing is not always good; it often leads to synchronized behavior.
- There are data problems in the industry, too; we do not have such an ideal model that contains all lines and their maximum loads.
- It is difficult to predict future developments; for example, our analytical estimates of the number of electric vehicles turned out to be too optimistic.

### In an analysis, on which location should one focus: a village, a city district, or a single transformer?

Are the transformers interconnected? Is there a backfeed / an influencing effect from the medium-voltage level?

### How else could the map UI be improved?

- Adjusting every household individually is too complex.
- Weights on households.
- Selection directly via "transformer"; it would make sense to have a map with clickable transformers from the very beginning.



## To which features the feedback was applied (also to which changes in prioritization the feedback led)

- Overload map (visualising overloads in the grid).
- Selecting individual transformers.
- Partially, the household configuration also derived from this feedback.
- Export of all dashboard data as JSON files, and of all tables as export files.
- Threshold values to determine whether the grid was underloaded or overloaded, with a short summary in the dashboard overview after the training.
- Simulation of alternative events -> different scenarios were considered useful.
- The dashboard now answers (derivably): which devices contributed most to the overload?
- Reprioritization of the cluster development: RLLib caused problems when distributing the workload across multiple nodes.
  - SLURM is not natively supported, only via a community guide; GridKIT ran on a single node only — to be regarded as a "stronger workstation computer"; the advantage of a cluster has not been exploited yet.
  - AZURE/AWS are not accessible to our project.

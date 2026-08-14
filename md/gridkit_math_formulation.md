# GridKIT — Mathematical Formulation

This document specifies the GridKIT decision problem from the math side: the multi-device
household environment, the per-device observation and the shared household reward of the RL
agents (Scenario 3), and the closed-form baseline behaviours compared in the study. Symbols map
directly to the code (`core/constants.py`, `core/models.py`, `grid_model/environment.py`,
`grid_model/device_profiles.py`, `grid_model/surrogate.py`, `grid_model/profiles.py`,
`scenarios/policies.py`, `rl_engine/`).

Each household hosts **three controllable devices** — an electric vehicle, a home battery and a
heat pump — plus an **exogenous** rooftop PV system. Every controllable device is its own agent;
all agents of the same device type share one policy. All quantities are per 24 h episode of $T$
discrete steps.

---

## 1. Constants and symbols

| Symbol | Meaning | Value | Source |
|---|---|---|---|
| $T$ | steps per episode | $96$ | `EPISODE_STEPS` |
| $\Delta t$ | step length (h) | $0.25$ | `TIMESTEP_HOURS` |
| $h_0$ | episode start hour | $12$ (noon) | `EPISODE_START_HOUR` |
| **EV** | | | |
| $\mathcal A^{\mathrm{ev}}$ | EV actions | $\{\mathrm{OFF,HALF,FULL}\}$ | `ChargingAction` |
| $P^{\mathrm{ev}}(a)$ | EV charge power (kW) | $\{0,\,3.7,\,7.4\}$ | `ACTION_TO_KW` |
| $x^\star$ | EV target SoC | $0.80$ | `EV_TARGET_SOC` |
| $B^{\mathrm{ev}}$ | EV battery (kWh) | $77$ | `EV_BATTERY_CAPACITY_KWH` |
| **Battery** | | | |
| $\mathcal A^{\mathrm{bat}}$ | battery actions | $\{\mathrm{DISCHARGE,IDLE,CHARGE}\}$ | `BatteryAction` |
| $P^{\mathrm{bat}}$ | battery rated power (kW) | $5$ | `BATTERY_MAX_POWER_KW` |
| $B^{\mathrm{bat}}$ | battery capacity (kWh) | $10$ | `BATTERY_CAPACITY_KWH` |
| $\eta$ | one-way efficiency | $0.95$ | `BATTERY_EFFICIENCY` |
| $[\underline s,\overline s]$ | usable SoC window | $[0.05,\,0.95]$ | `BATTERY_MIN/MAX_SOC` |
| **Heat pump** | | | |
| $\mathcal A^{\mathrm{hp}}$ | HP actions | $\{\mathrm{OFF,HEAT}\}$ | `HPAction` |
| $P^{\mathrm{hp}}$ | HP rated electric (kW) | $3$ | `HP_RATED_ELECTRIC_KW` |
| $\mathrm{COP}$ | coeff. of performance | $3.0$ | `HP_COP` |
| $C^{\mathrm{hp}}$ | thermal-buffer size (kWh) | $8$ | `HP_THERMAL_CAPACITY_KWH` |
| $\underline\tau$ | comfort floor (buffer SoC) | $0.30$ | `HP_COMFORT_MIN_SOC` |
| **PV / tariff** | | | |
| $\pi^{\mathrm{fit}}$ | feed-in tariff (€/kWh) | $0.08$ | `FEED_IN_TARIFF_EUR_KWH` |
| $P^{\mathrm{pv}}_{\mathrm{kwp}}$ | PV size (kWp) | $\sim\mathrm{U}(3,10)$ | `PV_PEAK_KWP_*` |
| **§14a / reward** | | | |
| $P_{\min}$ | §14a guaranteed floor (kW) | $4.2$ | `MIN_GUARANTEED_POWER_KW` |
| $w_c$ | cost weight | $0.1$ | `REWARD_ELECTRICITY_COST_WEIGHT` |
| $R^+,\,R^-$ | EV terminal bonus / penalty | $+10,\,-10$ | `REWARD_SOC_*` |
| $w_\tau$ | HP comfort penalty weight | $-10$ | `REWARD_HP_COMFORT_MISS_PENALTY` |
| $\gamma,\lambda,\varepsilon,\beta,c_v$ | PPO params | $0.99,0.95,0.2,0.01,0.5$ | `IPPO_*` |

Index $h \in \{1,\dots,H\}$ = active household, $d \in \{\mathrm{ev,bat,hp}\}$ = device
(one agent per $(h,d)$; agent id $=$ `f"{bus}::{d}"`), $t \in \{0,\dots,T-1\}$ = step. Wall-clock
hour of a step is $\mathrm{hr}(t) = (h_0 + t\,\Delta t)\bmod 24$.

---

## 2. Environment (partially observed stochastic game)

A Decentralized-POMDP / Markov game: $3H$ agents act simultaneously each step, share one physical
grid, and each observes only local information. **EV penetration** $\rho$ selects the active
households $H=\max(1,\mathrm{round}(\rho\,H_{\text{tot}}))$, spread evenly across the feeder; each
active household hosts the full device stack, the rest draw only base load.

### 2.1 Exogenous device profiles (GridCreator / pyCity)

Per episode each household $h$ receives a set of physics-based, weather-driven profiles sampled
from a cached annual pool (`grid_model/device_profiles.py`), built with GridCreator's pyCity
generators on bundled TRY weather. One noon→noon day is drawn feeder-wide (shared weather), then
each household draws its own occupancy sim; hourly series are resampled to the $T$ 15-min steps.

- $\ell_h(t)$ — inflexible base load (kW), pyCity stochastic **appliance + lighting** model
  (`create_appartment`, Richardson method). By construction this is the residential-electricity
  end-use only; EV, heat pump and PV are modelled as *separate* components to avoid double
  counting (see the base-load caveat in §8).
- $\mathrm{conn}_h(t)\in\{0,1\}$ — EV availability, occupancy-driven (`create_e_car`): the car is
  chargeable when someone is home. Its departure deadline is $d_h=\max\{t:\mathrm{conn}_h(t)=1\}$.
- $Q_h(t)$ — heat-pump **thermal demand** (kW), temperature/COP-driven (`create_hp`).
- $g_h(t)$ — PV generation (kW), weather + tilt/orientation driven (`create_pv`), $\propto$ kWp.
- $\theta(t)$ — outdoor temperature (°C), real TRY series (shared feeder-wide).

The day-ahead **price** $\pi(t)$ (€/kWh) remains a synthetic market signal (`profiles.price_profile`):
scenario multiplier $m_s\in\{0.7,1.0,1.4\}$ for LOW/MED/HIGH, morning + evening peaks and an
overnight trough (inside the EV connected window).

### 2.2 Actions and requested device power

Each device-agent chooses $a^d_h(t)\in\mathcal A^d$. Requested bus powers (kW), positive = load:

$$
r^{\mathrm{ev}}_h = \mathrm{conn}_h(t)\,\mathbb 1[x^{\mathrm{ev}}_h<x^\star]\,P^{\mathrm{ev}}(a^{\mathrm{ev}}_h),\qquad
r^{\mathrm{hp}}_h = P^{\mathrm{hp}}\,\mathbb 1[a^{\mathrm{hp}}_h=\mathrm{HEAT}],
$$

$$
r^{\mathrm{bat}}_h = \begin{cases}
+\min(P^{\mathrm{bat}},\,(\overline s-s_h)B^{\mathrm{bat}}/(\eta\Delta t)) & a^{\mathrm{bat}}_h=\mathrm{CHARGE}\\
-\min(P^{\mathrm{bat}},\,(s_h-\underline s)B^{\mathrm{bat}}\eta/\Delta t) & a^{\mathrm{bat}}_h=\mathrm{DISCHARGE}\\
0 & a^{\mathrm{bat}}_h=\mathrm{IDLE}
\end{cases}
$$

(battery charge/discharge are clipped to the usable SoC window and rated power). PV is not an
action — it is exogenous generation, auto-consumed by the meter (see §2.4); an independent
sell/self-consume PV action is degenerate when $\pi^{\mathrm{fit}}<\pi(t)$, so it is folded into
the battery's store-vs-export decision.

### 2.3 Household net power and power flow

The **net** active power injected at household $h$'s bus (positive = import from grid) is

$$
n_h(t) \;=\; \ell_h(t) + r^{\mathrm{ev}}_h(t) + r^{\mathrm{hp}}_h(t) + r^{\mathrm{bat}}_h(t) - g_h(t).
$$

Non-active households contribute $n_h=\ell_h$ only. The linearized radial surrogate
(`surrogate.RadialPowerFlow`, validated within ≈1.5 % of PyPSA `pf()`, ~3700× faster) maps the
per-bus nets $n_h$ to line/transformer loadings $u_e(t),u_{\mathrm{tr}}(t)$ (p.u.) and bus
voltages $v_b(t)$ (linearized DistFlow). PV export ($n_h<0$) is a valid reverse flow.

### 2.4 §14a curtailment (aggregate controllable draw)

Overload indicator (threshold $1$ p.u. on trafo and every line):
$\;O(t)=\mathbb 1[u_{\mathrm{tr}}>1 \lor \max_e u_e>1]$. §14a dims **controllable consumption**
— EV + HP + battery *charging* (discharge and PV are never curtailed). Let a household's
controllable request be $c_h=r^{\mathrm{ev}}_h+r^{\mathrm{hp}}_h+\max(0,r^{\mathrm{bat}}_h)$, and
$M(t)=\max(u_{\mathrm{tr}},\max_e u_e)$, scale $\alpha=1/M(t)<1$. If $O(t)=1$, the delivered
controllable power per household is proportionally dimmed with the guaranteed floor,

$$
\boxed{\;c^{\mathrm{del}}_h=\min\!\big(c_h,\;\max(\alpha\,c_h,\,P_{\min})\big)\;},\qquad
\phi_h=\frac{c^{\mathrm{del}}_h}{c_h},
$$

and each controllable component is scaled by $\phi_h$ (EV, HP, battery-charge); the grid is then
re-solved with the delivered nets. Otherwise $\phi_h=1$. §14a enters an agent's world **only**
through this reduced power — never as a reward term.

### 2.5 Device state dynamics

With delivered powers $p^{d,\mathrm{del}}_h=\phi_h r^d_h$:

$$
x^{\mathrm{ev}}_h(t{+}1)=\min\!\Big(1,\;x^{\mathrm{ev}}_h+\tfrac{p^{\mathrm{ev,del}}_h\Delta t}{B^{\mathrm{ev}}}\Big),
$$

$$
s_h(t{+}1)=\begin{cases}
s_h+\dfrac{p^{\mathrm{bat,del}}_h\,\eta\,\Delta t}{B^{\mathrm{bat}}} & \text{charging }(p^{\mathrm{bat}}_h>0)\\[6pt]
s_h-\dfrac{(|p^{\mathrm{bat}}_h|/\eta)\,\Delta t}{B^{\mathrm{bat}}} & \text{discharging }(p^{\mathrm{bat}}_h<0)
\end{cases}
$$

The heat pump is a **thermal buffer** (house inertia): running adds $P^{\mathrm{hp}}\!\cdot\!\mathrm{COP}$
thermal power, the weather demand $Q_h(t)$ drains it (clipped to $[0,1]$):

$$
\tau_h(t{+}1)=\mathrm{clip}\Big(\tau_h+\big(P^{\mathrm{hp}}\,\mathrm{COP}\,\mathbb 1[a^{\mathrm{hp}}_h=\mathrm{HEAT}]-Q_h(t)\,\mathrm{COP}\big)\tfrac{\Delta t}{C^{\mathrm{hp}}},\;0,\;1\Big).
$$

---

## 3. Observation (per device-agent)

Each agent observes a **uniform 10-vector** of local, smart-meter-plausible quantities
(`Observation.to_array_multidevice`); layout is identical across device types, but the first two
entries carry the agent's **own** device state. The two congestion terms (voltage, curtailment
ratio) and the net-load term are **lagged** by one step.

$$
o^d_h(t)=\Big[\underbrace{\sigma^d_h}_{\text{own SoC}},\;
\underbrace{u^d_h}_{\text{urgency}},\;\pi(t),\;\pi^{\mathrm{fit}},\;
\underbrace{n_h(t{-}1)}_{\text{net load}},\;g_h(t),\;\theta(t),\;
v_{b(h)}(t{-}1),\;\phi_h(t{-}1),\;\tfrac{t}{T}\Big]
$$

with own-SoC $\sigma^d_h=x^{\mathrm{ev}}_h/x^\star,\;s_h,\;\tau_h$ for ev/bat/hp, and urgency
$u^{\mathrm{ev}}_h=\max(0,(d_h-t)/T)$, $u^{\mathrm{bat}}_h=u^{\mathrm{hp}}_h=0$. Each field is
normalized to $[0,1]$ before the network (`rl_engine/obs_norm.normalize_observation_multidevice`;
signed net load mapped by $\tfrac{n+15}{30}$) then clipped.

---

## 4. Reward (Scenario 3 — the RL agents)

**All three device-agents at a household share one household-net reward** — minimise the net
electricity bill, meet the EV SoC target, keep the heat pump comfortable
(`grid_model/environment.py::_reward`). With the household net $n_h(t)$ (delivered):

$$
\boxed{\;
R_h(t) = -\,w_c\big[\pi(t)\,n_h^+ - \pi^{\mathrm{fit}}\,n_h^-\big]\Delta t
\;+\; w_\tau\,\max(0,\underline\tau-\tau_h)
\;+\; \mathbb 1[t=d_h]\cdot\!\begin{cases}R^+&x^{\mathrm{ev}}_h\!\ge x^\star\\ R^-&\text{else}\end{cases}\;}
$$

where $n_h^+=\max(0,n_h)$ is grid import (paid at retail $\pi$) and $n_h^-=\max(0,-n_h)$ is
export (paid at feed-in $\pi^{\mathrm{fit}}$). The middle term is a **per-step** comfort penalty
proportional to how far the thermal buffer sits below the floor (prevents "off all day then a
terminal blast"). Because $\pi^{\mathrm{fit}}<\pi$, self-consuming PV surplus (via the battery /
flexible loads) beats exporting it — the coupling the shared reward is meant to teach. §14a is
felt only indirectly: dimming lowers delivered power, risking the EV miss and comfort deficits.
The $P_{\min}$ floor means a tightly-scheduled household can still rationally charge *through*
congestion.

---

## 5. RL objective (Independent PPO, one shared policy per device type)

There are **three** shared stochastic policies $\pi_{\theta_d}(a\mid\tilde o)$ — `ev_policy`,
`battery_policy`, `hp_policy` (`rl_engine/ippo_config.py`) — each a categorical over its device's
action space ($|\mathcal A^{\mathrm{ev}}|=|\mathcal A^{\mathrm{bat}}|=3,\;|\mathcal A^{\mathrm{hp}}|=2$).
Agent $(h,d)$ maps to policy $d$ (`policy_mapping_fn` by agent-id suffix); this **agent-per-device**
factorization keeps each action space small instead of a $3\!\cdot\!3\!\cdot\!2=18$-way joint. Each
agent maximizes its (shared household) discounted return $J=\mathbb E[\sum_t\gamma^t R_h(t)]$ via
the clipped PPO surrogate with GAE ($\gamma=0.99,\lambda=0.95,\varepsilon=0.2$, entropy $\beta=0.01$,
value weight $c_v=0.5$, lr $3\times10^{-4}$). At run time each agent samples independently
(`rl_policy.RLlibPolicyAdapter`, per-device batched inference), so identical agents scatter in time
rather than re-synchronizing.

---

## 6. Baseline behaviour models

All scenarios run on the identical environment (§2) and are scored by the same metrics (§7). A
baseline sets the **EV** leg; the battery and heat pump use fixed rule-based controllers (the
status quo of real home energy management), so scenarios stay comparable to the learned policy
(`scenarios/policies.py`).

**Battery — greedy self-consumption** (both scenarios): CHARGE when PV surplus is available
($g_h>0.1$ kW, $s_h<\overline s$), DISCHARGE while importing ($n_h>0,\;s_h>\underline s$), else IDLE.

**Heat pump — thermostatic** (both scenarios): HEAT while $\tau_h<0.6$, else OFF (ignores price/grid).

### 6.1 Scenario 1 — flat tariff, immediate EV charging

$a^{\mathrm{ev}}_h(t)=\mathrm{FULL}$ if $\mathrm{conn}_h(t)=1\wedge x^{\mathrm{ev}}_h<x^\star$, else OFF.
Uncoordinated, clustered by arrival and coincident with the evening base-load peak.

### 6.2 Scenario 2 — naïve price-following EV

From the published day-ahead prices, pick the cheapest contiguous window long enough to finish the
remaining energy $E_h=\max(0,\,x^\star-x^{\mathrm{ev}}_h)\,B^{\mathrm{ev}}$ (window length
$\lceil E_h/(P^{\mathrm{ev}}_{\mathrm{FULL}}\Delta t)\rceil$), add start-time jitter
$\xi\sim\mathcal N(0,\sigma^2)$, then charge full until done. $\sigma$ is the
manual↔automated knob: $\sigma=8$ (≈2 h) = human spread; $\sigma\to0$ = app-driven, every household
charges at the *same* cheapest step — the synchronization that stresses §14a (now amplified because
price-reactive homes also shift battery/HP into the same window).

### 6.3 Scenario 3 — selfish, congestion-aware RL

The learned per-device stochastic policies of §4–5, evaluated through the same runner. Unlike 1–2,
the input includes the lagged local voltage, curtailment ratio and net load, giving self-interest a
channel to sense and avoid congestion.

---

## 7. Evaluation metrics

Per episode, aggregated as mean ± std over seeds (`scenarios/runner.py`):

$$
\textbf{Curtailment events}=\sum_t O(t),\qquad
\textbf{EV SoC satisfaction}=\tfrac1H\sum_h \mathbb 1[x^{\mathrm{ev}}_h(d_h)\ge x^\star],
$$

$$
\textbf{Transformer peak}=\max_t u_{\mathrm{tr}}(t),\qquad
\textbf{Mean return}=\tfrac1{3H}\sum_{h,d}\sum_t R_h(t).
$$

---

## 8. Modelling assumptions (scope)

Loss-free linear power flow; active power only ($Q\approx0$); synthetic price shape; real TRY
weather but a single climate day drawn feeder-wide; one shared policy per device type; feeder-level
(not line-localized) curtailment; PV must-take (self-consumption via net meter, no explicit
curtailment); heat pump as a single-buffer thermostatic load with a seasonal-average COP; battery
one-way efficiency on each leg. The surrogate is the only physics backend in the multi-device env
(the PyPSA validation path applies to the EV-only model). Results are a **relative** comparison of a
demonstrated mechanism under these named simplifications — not a calibrated forecast for a real feeder.

### 8.1 Base-load double-counting caveat

The base load $\ell_h$ and the controllable devices are additive, so it matters whether $\ell_h$
already contains any of the device end-uses. The pyCity Richardson appliance set (`create_appartment`,
`randomize_appliances=True`) is drawn per household by appliance penetration and, by design, covers
domestic appliances + lighting — **not** EV charging or PV. Two residual overlaps with the heat pump
exist but are minor and intentionally left in:

- **Electric space heating** (storage heaters + "other electric space heating") is the only end-use
  that genuinely overlaps our space-heating HP. In the Richardson set it appears in only
  **≈ 2.6–2.8 %** of households (a UK-derived model where most homes heat with gas), so the overlap
  with $r^{\mathrm{hp}}$ is a small residual, not a systematic double count.
- **Electric water heating** (electric shower ≈ 67 %, instantaneous/DE water heaters ≈ 17 %) is
  present in the base load, but it is domestic hot water — a *different* end-use than our
  space-heating HP ($Q_h$ is temperature/COP-driven space heat, active below 15 °C) — so it is not a
  double count of $r^{\mathrm{hp}}$.

Crucially, $\ell_h$ is an **identical exogenous input across all scenarios** (1, 2, 3): any residual
overlap shifts the absolute load level equally for every policy and therefore **cannot bias the
relative comparison**, which is the study's actual claim. It only slightly affects absolute realism.
(If exactness is required, the two electric-space-heating appliances can be zeroed in the base-load
generation, since that end-use is modelled explicitly as the heat pump.)

import argparse
import csv
import math
import random
import time
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    HAS_MPL = True
except Exception:
    HAS_MPL = False

REQ = [
    "attendance_rate",
    "missing_submissions",
    "avg_quiz_score",
    "lms_activity",
    "study_hours_per_week",
    "days_to_deadline",
]


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass(frozen=True)
class State:
    attendance_rate: float
    missing_submissions: int
    avg_quiz_score: float
    lms_activity: float
    study_hours_per_week: float
    days_to_deadline: int
    fatigue: float = 18.0
    extra_study: float = 0.0
    steps: int = 0


@dataclass
class Constraints:
    study_limit_per_day: float = 2.0
    tutor_available: bool = True
    max_steps: int = 20
    goal_risk: float = 34.0
    deadline_cost_weight: float = 0.9
    fatigue_cost_weight: float = 0.15
    initial_days: int = 7

    @property
    def max_extra_study(self) -> float:
        return max(1, self.initial_days) * self.study_limit_per_day


@dataclass
class Result:
    method: str
    found: bool
    cost: float
    actions: List[str]
    final_state: State
    final_risk: float
    runtime: float
    nodes: int


@dataclass
class Node:
    state: State
    g: float
    h: float
    action: Optional[str]
    parent: Optional["Node"]

    @property
    def f(self) -> float:
        return self.g + self.h


class Planner:
    target_att, target_quiz, target_lms = 75.0, 65.0, 55.0

    def __init__(self, init: State, c: Constraints):
        self.init = init
        self.c = c
        self.actions = [
            "Attend_Class",
            "Study_1_Hour",
            "Submit_Assignment",
            "Practice_Quiz",
            "Meet_Tutor",
            "Rest",
        ]
        self.impl: Dict[str, Callable[[State], Optional[Tuple[State, float]]]] = {
            "Attend_Class": self._attend,
            "Study_1_Hour": self._study,
            "Submit_Assignment": self._submit,
            "Practice_Quiz": self._practice,
            "Meet_Tutor": self._tutor,
            "Rest": self._rest,
        }

    def risk(self, s: State) -> float:
        missing = min(s.missing_submissions, 6) / 6 * 100
        attendance = 100 - s.attendance_rate
        quiz = 100 - s.avg_quiz_score
        lms = 100 - s.lms_activity
        study_def = max(0.0, 8.0 - s.study_hours_per_week) * 12.5
        deadline = 100.0 if s.days_to_deadline <= 0 else clamp((14 - s.days_to_deadline) / 14 * 100, 0, 100)
        score = (
            0.22 * attendance
            + 0.24 * missing
            + 0.20 * quiz
            + 0.13 * lms
            + 0.07 * deadline
            + 0.11 * s.fatigue
            + 0.03 * study_def
        )
        return clamp(score, 0.0, 100.0)

    def goal(self, s: State) -> bool:
        return (
            s.missing_submissions == 0
            and s.attendance_rate >= self.target_att
            and s.avg_quiz_score >= self.target_quiz
            and self.risk(s) <= self.c.goal_risk
        )

    def heuristic(self, s: State) -> float:
        a = max(0.0, self.target_att - s.attendance_rate) / 4.0 * 0.8
        q = max(0.0, self.target_quiz - s.avg_quiz_score) / 5.0 * 1.0
        l = max(0.0, self.target_lms - s.lms_activity) / 3.0 * 0.7
        m = s.missing_submissions * 1.0
        r = max(0.0, self.risk(s) - self.c.goal_risk) * 0.05
        return max(a, q, l, m, r)

    def key(self, s: State):
        return (
            round(s.attendance_rate, 1),
            s.missing_submissions,
            round(s.avg_quiz_score, 1),
            round(s.lms_activity, 1),
            round(s.study_hours_per_week, 1),
            s.days_to_deadline,
            round(s.fatigue, 1),
            round(s.extra_study, 1),
        )

    def search(self, method: str) -> Result:
        method = method.lower()
        t0 = time.perf_counter()
        first = Node(self.init, 0.0, self.heuristic(self.init), None, None)
        h: List[Tuple[float, int, Node]] = []
        tie = count()
        heappush(h, (self._prio(first, method), next(tie), first))
        best = {self.key(first.state): 0.0}
        best_node = first
        nodes = 0

        while h:
            _, _, cur = heappop(h)
            k = self.key(cur.state)
            if cur.g > best.get(k, math.inf) + 1e-9:
                continue
            nodes += 1
            if self.risk(cur.state) < self.risk(best_node.state):
                best_node = cur
            if self.goal(cur.state):
                return self._result(method, True, cur, nodes, t0)
            if cur.state.steps >= self.c.max_steps:
                continue
            for a in self.actions:
                nxt = self.impl[a](cur.state)
                if nxt is None:
                    continue
                ns, c = nxt
                ng = cur.g + c
                nk = self.key(ns)
                if ng + 1e-9 >= best.get(nk, math.inf):
                    continue
                best[nk] = ng
                node = Node(ns, ng, self.heuristic(ns), a, cur)
                heappush(h, (self._prio(node, method), next(tie), node))

        return self._result(method, False, best_node, nodes, t0)

    def _result(self, method: str, found: bool, n: Node, nodes: int, t0: float) -> Result:
        actions: List[str] = []
        cur = n
        while cur and cur.action:
            actions.append(cur.action)
            cur = cur.parent
        actions.reverse()
        return Result(method.upper(), found, n.g, actions, n.state, self.risk(n.state), time.perf_counter() - t0, nodes)

    def _prio(self, n: Node, method: str) -> float:
        if method == "astar":
            return n.f
        if method == "ucs":
            return n.g
        return n.h

    def _cost(self, name: str, s: State, ns: State) -> float:
        base = {
            "Attend_Class": 2.0,
            "Study_1_Hour": 1.6,
            "Submit_Assignment": 2.8,
            "Practice_Quiz": 2.1,
            "Meet_Tutor": 3.5,
            "Rest": 1.2,
        }[name]
        urgency = max(0, 4 - s.days_to_deadline) * self.c.deadline_cost_weight
        fatigue = max(0.0, s.fatigue - 30.0) * self.c.fatigue_cost_weight
        overdue = max(0, -ns.days_to_deadline) * 3.0
        risk_pen = 1.0 if self.risk(ns) > self.risk(s) else 0.0
        if name == "Rest" and s.days_to_deadline <= 2:
            risk_pen += 1.5
        return base + urgency + fatigue + overdue + risk_pen

    def _step(
        self,
        s: State,
        name: str,
        att: float = 0.0,
        miss: int = 0,
        quiz: float = 0.0,
        lms: float = 0.0,
        study: float = 0.0,
        fatigue: float = 0.0,
        day: int = -1,
    ) -> Tuple[State, float]:
        ns = State(
            attendance_rate=clamp(s.attendance_rate + att, 0, 100),
            missing_submissions=max(0, s.missing_submissions + miss),
            avg_quiz_score=clamp(s.avg_quiz_score + quiz, 0, 100),
            lms_activity=clamp(s.lms_activity + lms, 0, 100),
            study_hours_per_week=clamp(s.study_hours_per_week + study, 0, 60),
            days_to_deadline=s.days_to_deadline + day,
            fatigue=clamp(s.fatigue + fatigue, 0, 100),
            extra_study=s.extra_study + max(0.0, study),
            steps=s.steps + 1,
        )
        return ns, self._cost(name, s, ns)

    def _attend(self, s: State):
        return self._step(s, "Attend_Class", att=4.0, lms=2.5, fatigue=5.0)

    def _study(self, s: State):
        if s.extra_study + 1.0 > self.c.max_extra_study:
            return None
        return self._step(s, "Study_1_Hour", study=1.0, quiz=2.0, lms=1.0, fatigue=8.0)

    def _submit(self, s: State):
        if s.missing_submissions <= 0:
            return None
        return self._step(s, "Submit_Assignment", miss=-1, lms=3.0, fatigue=5.0)

    def _practice(self, s: State):
        if s.extra_study + 0.5 > self.c.max_extra_study:
            return None
        return self._step(s, "Practice_Quiz", quiz=4.0, study=0.5, lms=1.0, fatigue=7.0)

    def _tutor(self, s: State):
        if not self.c.tutor_available:
            return None
        return self._step(s, "Meet_Tutor", att=1.0, quiz=5.0, lms=2.0, fatigue=4.0)

    def _rest(self, s: State):
        if s.fatigue <= 24.0:
            return None
        # Rest is modeled as a short recovery block: no deadline day lost, but effort still has a cost.
        return self._step(s, "Rest", fatigue=-18.0, day=0)


def parse_csv(path: Path) -> List[Dict[str, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("CSV is empty")
    out: List[Dict[str, float]] = []
    for i, r in enumerate(rows):
        n = {k.strip().lower(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()}
        miss = [x for x in REQ if x not in n]
        if miss:
            raise ValueError("Missing columns: " + ", ".join(miss))
        out.append(
            {
                "student_id": n.get("student_id", f"S{i+1:03d}"),
                "attendance_rate": float(n["attendance_rate"]),
                "missing_submissions": int(float(n["missing_submissions"])),
                "avg_quiz_score": float(n["avg_quiz_score"]),
                "lms_activity": float(n["lms_activity"]),
                "study_hours_per_week": float(n["study_hours_per_week"]),
                "days_to_deadline": int(float(n["days_to_deadline"])),
            }
        )
    return out


def estimate_initial_fatigue(r: Dict[str, float]) -> float:
    # Higher study load and backlog imply more fatigue; more days left reduces pressure.
    raw = (
        16.0
        + 1.8 * float(r["study_hours_per_week"])
        + 2.0 * float(r["missing_submissions"])
        - 0.7 * float(r["days_to_deadline"])
    )
    return clamp(raw, 12.0, 85.0)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Student At-Risk Recovery Planner (A*)")
        self.geometry("1280x900")

        self.data: List[Dict[str, float]] = []
        self.lookup: Dict[str, Dict[str, float]] = {}

        self.study_limit = tk.DoubleVar(value=2.0)
        self.deadline_shift = tk.IntVar(value=0)
        self.tutor_ok = tk.BooleanVar(value=True)
        self.max_steps = tk.IntVar(value=20)
        self.goal_risk = tk.DoubleVar(value=34.0)
        self.deadline_cost_weight = tk.DoubleVar(value=0.9)
        self.fatigue_cost_weight = tk.DoubleVar(value=0.15)
        self.student = tk.StringVar(value="")
        self.input_mode = tk.StringVar(value="custom")
        self.in_attendance = tk.StringVar(value="")
        self.in_missing = tk.StringVar(value="")
        self.in_quiz = tk.StringVar(value="")
        self.in_lms = tk.StringVar(value="")
        self.in_study = tk.StringVar(value="")
        self.in_days = tk.StringVar(value="")
        self.in_fatigue = tk.StringVar(value="")
        self._refresh_after_id: Optional[str] = None

        self._ui()
        self._bind_auto_refresh()
        p = Path(__file__).resolve().parent / "STUDEN_1.CSV"
        if p.exists():
            self._load(p, show_message=False)

    def _bind_auto_refresh(self):
        for var in (
            self.study_limit,
            self.deadline_shift,
            self.tutor_ok,
            self.max_steps,
            self.goal_risk,
            self.deadline_cost_weight,
            self.fatigue_cost_weight,
            self.input_mode,
            self.in_attendance,
            self.in_missing,
            self.in_quiz,
            self.in_lms,
            self.in_study,
            self.in_days,
            self.in_fatigue,
        ):
            var.trace_add("write", lambda *_: self._schedule_refresh())

    def _ui(self):
        top = ttk.Frame(self, padding=8)
        top.pack(fill=tk.X)
        ttk.Button(top, text="Load CSV", command=self.load_csv).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Generate Scenarios", command=self.gen_data).pack(side=tk.LEFT, padx=4)
        ttk.Label(top, text="Student:").pack(side=tk.LEFT, padx=(12, 4))
        self.combo = ttk.Combobox(top, textvariable=self.student, state="readonly", width=18)
        self.combo.pack(side=tk.LEFT)
        self.combo.bind("<<ComboboxSelected>>", self._on_student_change)
        ttk.Label(top, text="Input source:").pack(side=tk.LEFT, padx=(12, 4))
        ttk.Radiobutton(top, text="CSV", variable=self.input_mode, value="file", command=self._on_source_change).pack(
            side=tk.LEFT
        )
        ttk.Radiobutton(
            top, text="Custom", variable=self.input_mode, value="custom", command=self._on_source_change
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(top, text="Copy Student -> Custom", command=self._copy_selected_to_custom).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Run Planner", command=self.run).pack(side=tk.LEFT, padx=8)
        ttk.Button(top, text="Run 3 What-if Cases", command=self.run_whatif).pack(side=tk.LEFT, padx=4)

        sbox = ttk.LabelFrame(self, text="Scenario Inputs", padding=8)
        sbox.pack(fill=tk.X, padx=8, pady=(0, 8))
        self.custom_widgets: List[tk.Widget] = []

        def add_input(row: int, col: int, label: str, var: tk.Variable):
            ttk.Label(sbox, text=label).grid(row=row, column=col * 2, padx=4, pady=3, sticky=tk.W)
            entry = ttk.Entry(sbox, textvariable=var, width=9)
            entry.grid(row=row, column=col * 2 + 1, padx=(0, 12), pady=3, sticky=tk.W)
            self.custom_widgets.append(entry)

        add_input(0, 0, "Attendance (0-100):", self.in_attendance)
        add_input(0, 1, "Missing submissions:", self.in_missing)
        add_input(0, 2, "Avg quiz (0-100):", self.in_quiz)
        add_input(0, 3, "LMS activity (0-100):", self.in_lms)
        add_input(1, 0, "Study hours/week:", self.in_study)
        add_input(1, 1, "Days to deadline:", self.in_days)
        add_input(1, 2, "Initial fatigue (0-100):", self.in_fatigue)
        ttk.Label(
            sbox,
            text="Custom fields are used when Input source = Custom. CSV mode uses the selected student row.",
            foreground="#2b3e5f",
        ).grid(row=2, column=0, columnspan=8, sticky=tk.W, padx=4, pady=(4, 0))

        c = ttk.LabelFrame(self, text="What-if Constraints", padding=8)
        c.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Label(c, text="Study limit/day:").grid(row=0, column=0, padx=4)
        ttk.Entry(c, textvariable=self.study_limit, width=8).grid(row=0, column=1, padx=4)
        ttk.Label(c, text="Deadline shift(days):").grid(row=0, column=2, padx=4)
        ttk.Entry(c, textvariable=self.deadline_shift, width=8).grid(row=0, column=3, padx=4)
        ttk.Label(c, text="Max steps:").grid(row=0, column=4, padx=4)
        ttk.Entry(c, textvariable=self.max_steps, width=8).grid(row=0, column=5, padx=4)
        ttk.Checkbutton(c, text="Tutor available", variable=self.tutor_ok).grid(row=0, column=6, padx=8)
        ttk.Label(c, text="Risk threshold(goal):").grid(row=1, column=0, padx=4, pady=(6, 0))
        ttk.Entry(c, textvariable=self.goal_risk, width=8).grid(row=1, column=1, padx=4, pady=(6, 0))
        ttk.Label(c, text="Deadline cost weight:").grid(row=1, column=2, padx=4, pady=(6, 0))
        ttk.Entry(c, textvariable=self.deadline_cost_weight, width=8).grid(row=1, column=3, padx=4, pady=(6, 0))
        ttk.Label(c, text="Fatigue cost weight:").grid(row=1, column=4, padx=4, pady=(6, 0))
        ttk.Entry(c, textvariable=self.fatigue_cost_weight, width=8).grid(row=1, column=5, padx=4, pady=(6, 0))

        main = ttk.Frame(self, padding=8)
        main.pack(fill=tk.BOTH, expand=True)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=3)
        main.rowconfigure(1, weight=2)

        left = ttk.LabelFrame(main, text="Plan and Metrics", padding=8)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))
        self.summary = tk.StringVar(value="Load data and select a student.")
        ttk.Label(left, textvariable=self.summary, foreground="#173d6d").pack(anchor=tk.W, pady=(0, 6))
        self.metrics = tk.StringVar(value="Run planner to see metrics.")
        ttk.Label(left, textvariable=self.metrics, justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 6))
        self.plan = tk.Text(left, height=24, wrap=tk.WORD)
        self.plan.pack(fill=tk.BOTH, expand=True)

        right = ttk.LabelFrame(main, text="Plots", padding=8)
        right.grid(row=0, column=1, sticky="nsew", pady=(0, 8))
        if HAS_MPL:
            self.fig = Figure(figsize=(6.2, 4.8), dpi=100)
            self.ax_risk = self.fig.add_subplot(121)
            self.ax_actions = self.fig.add_subplot(122)
            self.canvas = FigureCanvasTkAgg(self.fig, master=right)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self._draw_empty()
        else:
            ttk.Label(right, text="Install matplotlib for charts.", foreground="#7f1d1d").pack(anchor=tk.W)

        lower = ttk.LabelFrame(main, text="Baseline Comparison", padding=8)
        lower.grid(row=1, column=0, columnspan=2, sticky="nsew")
        cols = ("method", "found", "cost", "runtime_ms", "nodes", "final_risk", "plan_len")
        self.table = ttk.Treeview(lower, columns=cols, show="headings", height=6)
        for col in cols:
            self.table.heading(col, text=col)
            self.table.column(col, width=150, anchor=tk.CENTER)
        self.table.pack(fill=tk.BOTH, expand=True)

        wbox = ttk.LabelFrame(self, text="What-if Results (A*)", padding=8)
        wbox.pack(fill=tk.BOTH, padx=8, pady=(0, 8))
        wcols = ("case", "found", "cost", "final_risk", "nodes", "runtime_ms")
        self.wtable = ttk.Treeview(wbox, columns=wcols, show="headings", height=4)
        for col in wcols:
            self.wtable.heading(col, text=col)
            self.wtable.column(col, width=170, anchor=tk.CENTER)
        self.wtable.pack(fill=tk.BOTH, expand=True)
        self._update_custom_input_state()

    def _draw_empty(self):
        if not HAS_MPL:
            return
        self.ax_risk.clear()
        self.ax_actions.clear()
        self.ax_risk.set_title("Risk Before vs After")
        self.ax_risk.bar(["Before", "After"], [0, 0], color=["#d65f5f", "#5f9ed6"])
        self.ax_risk.set_ylim(0, 100)
        self.ax_actions.set_title("Action Counts")
        self.ax_actions.bar(["N/A"], [0], color="#83b66f")
        self.fig.tight_layout()
        self.canvas.draw()

    def load_csv(self):
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv;*.CSV"), ("All", "*.*")])
        if p:
            self._load(Path(p))

    def _load(self, p: Path, show_message: bool = True):
        try:
            rows = parse_csv(p)
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            return
        self.data = rows
        self.lookup = {r["student_id"]: r for r in rows}
        ids = list(self.lookup.keys())
        self.combo["values"] = ids
        if ids:
            self.student.set(ids[0])
            self._show_student()
            self._schedule_refresh()
        if show_message:
            messagebox.showinfo("Loaded", f"Loaded {len(rows)} records from {p}")

    def gen_data(self):
        rows = []
        for i in range(1, 31):
            rows.append(
                {
                    "student_id": f"G{i:03d}",
                    "attendance_rate": round(random.uniform(45, 96), 1),
                    "missing_submissions": random.randint(0, 6),
                    "avg_quiz_score": round(random.uniform(40, 92), 1),
                    "lms_activity": round(random.uniform(20, 95), 1),
                    "study_hours_per_week": round(random.uniform(0, 16), 1),
                    "days_to_deadline": random.randint(1, 14),
                }
            )
        self.data = rows
        self.lookup = {r["student_id"]: r for r in rows}
        self.combo["values"] = list(self.lookup.keys())
        self.student.set(rows[0]["student_id"])
        self._show_student()
        self._schedule_refresh()
        messagebox.showinfo("Generated", "Generated 30 synthetic scenarios.")

    def _selected(self) -> Optional[Dict[str, float]]:
        return self.lookup.get(self.student.get().strip())

    def _show_student(self):
        if self.input_mode.get() == "custom":
            try:
                if self._any_custom_field_blank():
                    self.summary.set("Custom mode active. Enter your own values in Scenario Inputs.")
                    return
                self.summary.set(
                    f"Custom | attendance={float(self.in_attendance.get().strip()):.1f}% | "
                    f"missing={int(float(self.in_missing.get().strip()))} | "
                    f"quiz={float(self.in_quiz.get().strip()):.1f} | "
                    f"lms={float(self.in_lms.get().strip()):.1f} | "
                    f"study={float(self.in_study.get().strip()):.1f}h | "
                    f"deadline={int(float(self.in_days.get().strip()))}d | "
                    f"fatigue={float(self.in_fatigue.get().strip()):.1f}"
                )
            except Exception:
                self.summary.set("Custom mode active. Enter valid numeric scenario inputs.")
            return
        r = self._selected()
        if not r:
            self.summary.set("No student selected.")
            return
        self.summary.set(
            f"{r['student_id']} | attendance={r['attendance_rate']:.1f}% | missing={r['missing_submissions']} | "
            f"quiz={r['avg_quiz_score']:.1f} | lms={r['lms_activity']:.1f} | "
            f"study={r['study_hours_per_week']:.1f}h | deadline={r['days_to_deadline']}d"
        )

    def _on_student_change(self, _event=None):
        self._show_student()
        self._schedule_refresh()

    def _on_source_change(self):
        self._update_custom_input_state()
        self._show_student()
        if self.input_mode.get() == "custom" and self._any_custom_field_blank():
            self._clear_outputs("Enter custom scenario inputs, then run planner.")
        self._schedule_refresh()

    def _update_custom_input_state(self):
        state = "normal" if self.input_mode.get() == "custom" else "disabled"
        for w in getattr(self, "custom_widgets", []):
            w.configure(state=state)

    def _copy_selected_to_custom(self, show_warnings: bool = True):
        r = self._selected()
        if not r:
            if show_warnings:
                messagebox.showwarning("No student", "Select a student row first to copy values.")
            return
        self.in_attendance.set(f"{float(r['attendance_rate']):.1f}")
        self.in_missing.set(str(int(r["missing_submissions"])))
        self.in_quiz.set(f"{float(r['avg_quiz_score']):.1f}")
        self.in_lms.set(f"{float(r['lms_activity']):.1f}")
        self.in_study.set(f"{float(r['study_hours_per_week']):.1f}")
        self.in_days.set(str(int(r["days_to_deadline"])))
        self.in_fatigue.set(f"{estimate_initial_fatigue(r):.1f}")
        self.input_mode.set("custom")
        self._update_custom_input_state()

    def _custom_field_specs(self):
        return [
            ("Attendance", self.in_attendance.get(), 0.0, 100.0, False),
            ("Missing submissions", self.in_missing.get(), 0.0, 20.0, True),
            ("Avg quiz score", self.in_quiz.get(), 0.0, 100.0, False),
            ("LMS activity", self.in_lms.get(), 0.0, 100.0, False),
            ("Study hours/week", self.in_study.get(), 0.0, 60.0, False),
            ("Days to deadline", self.in_days.get(), 0.0, 365.0, True),
            ("Initial fatigue", self.in_fatigue.get(), 0.0, 100.0, False),
        ]

    def _any_custom_field_blank(self) -> bool:
        for _, raw, *_ in self._custom_field_specs():
            if str(raw).strip() == "":
                return True
        return False

    def _parse_custom_value(
        self, label: str, raw: str, low: float, high: float, as_int: bool = False
    ) -> float:
        text = str(raw).strip()
        if text == "":
            raise ValueError(f"{label} is required.")
        value = float(text)
        value = clamp(value, low, high)
        if as_int:
            return int(round(value))
        return value

    def _clear_outputs(self, message: str = "Run planner to see metrics."):
        self.metrics.set(message)
        self.plan.configure(state=tk.NORMAL)
        self.plan.delete("1.0", tk.END)
        self.plan.insert(tk.END, "No plan generated.\n")
        self.plan.configure(state=tk.DISABLED)
        for i in self.table.get_children():
            self.table.delete(i)
        for i in self.wtable.get_children():
            self.wtable.delete(i)
        if HAS_MPL:
            self._draw_empty()

    def _schedule_refresh(self):
        if self._refresh_after_id:
            self.after_cancel(self._refresh_after_id)
        self._refresh_after_id = self.after(250, self._auto_refresh)

    def _auto_refresh(self):
        self._refresh_after_id = None
        if self.input_mode.get() == "file" and not self._selected():
            return
        if self.input_mode.get() == "custom" and self._any_custom_field_blank():
            self._show_student()
            self._clear_outputs("Enter custom scenario inputs, then run planner.")
            return
        self._show_student()
        self.run(show_warnings=False, refresh_whatif=True)

    def _state(self, r: Dict[str, float], shift: int = 0) -> State:
        adjusted_days = max(0, int(r["days_to_deadline"]) + shift)
        fatigue = estimate_initial_fatigue(
            {
                "study_hours_per_week": r["study_hours_per_week"],
                "missing_submissions": r["missing_submissions"],
                "days_to_deadline": adjusted_days,
            }
        )
        return State(
            attendance_rate=float(r["attendance_rate"]),
            missing_submissions=int(r["missing_submissions"]),
            avg_quiz_score=float(r["avg_quiz_score"]),
            lms_activity=float(r["lms_activity"]),
            study_hours_per_week=float(r["study_hours_per_week"]),
            days_to_deadline=adjusted_days,
            fatigue=fatigue,
        )

    def _state_from_custom(self, shift: int = 0) -> State:
        parsed: Dict[str, float] = {}
        for label, raw, low, high, as_int in self._custom_field_specs():
            parsed[label] = self._parse_custom_value(label, raw, low, high, as_int)
        adjusted_days = max(0, int(parsed["Days to deadline"]) + shift)
        return State(
            attendance_rate=float(parsed["Attendance"]),
            missing_submissions=int(parsed["Missing submissions"]),
            avg_quiz_score=float(parsed["Avg quiz score"]),
            lms_activity=float(parsed["LMS activity"]),
            study_hours_per_week=float(parsed["Study hours/week"]),
            days_to_deadline=adjusted_days,
            fatigue=float(parsed["Initial fatigue"]),
        )

    def _state_from_source(self, shift: int = 0, show_warnings: bool = True) -> Optional[State]:
        if self.input_mode.get() == "custom":
            try:
                return self._state_from_custom(shift=shift)
            except Exception as e:
                if show_warnings:
                    messagebox.showerror("Input error", f"Invalid custom scenario values.\n{e}")
                return None
        r = self._selected()
        if not r:
            if show_warnings:
                messagebox.showwarning("No student", "Load data and select a student.")
            return None
        return self._state(r, shift=shift)

    def _constraints(self, days: int, study: Optional[float] = None, tutor: Optional[bool] = None) -> Constraints:
        study_limit_value = float(self.study_limit.get() if study is None else study)
        steps_value = int(float(self.max_steps.get()))
        return Constraints(
            study_limit_per_day=max(0.0, study_limit_value),
            tutor_available=bool(self.tutor_ok.get() if tutor is None else tutor),
            max_steps=max(1, steps_value),
            goal_risk=clamp(float(self.goal_risk.get()), 0.0, 100.0),
            deadline_cost_weight=max(0.0, float(self.deadline_cost_weight.get())),
            fatigue_cost_weight=max(0.0, float(self.fatigue_cost_weight.get())),
            initial_days=max(1, days),
        )

    def run(self, show_warnings: bool = True, refresh_whatif: bool = False) -> bool:
        try:
            shift = int(self.deadline_shift.get())
            s0 = self._state_from_source(shift=shift, show_warnings=show_warnings)
            if s0 is None:
                if not show_warnings:
                    self._clear_outputs("Enter valid inputs to generate a plan.")
                return False
            c = self._constraints(s0.days_to_deadline)
            p = Planner(s0, c)
            before = p.risk(s0)
            res = [p.search("astar"), p.search("ucs"), p.search("greedy")]
        except Exception as e:
            if show_warnings:
                messagebox.showerror("Input error", str(e))
            else:
                self._clear_outputs("Enter valid inputs to generate a plan.")
            return False
        self._show_result(s0, before, res[0])
        self._show_table(res)
        self._show_plots(before, res[0])
        if refresh_whatif:
            self.run_whatif(show_warnings=False, show_done=False)
        return True

    def _show_result(self, initial_state: State, before: float, r: Result):
        status = "Not At-Risk" if r.found else "Partially Improved"
        self.metrics.set(
            f"Initial risk: {before:.2f}\n"
            f"Initial state -> attendance={initial_state.attendance_rate:.1f}, missing={initial_state.missing_submissions}, "
            f"quiz={initial_state.avg_quiz_score:.1f}, lms={initial_state.lms_activity:.1f}, "
            f"study={initial_state.study_hours_per_week:.1f}, deadline={initial_state.days_to_deadline}, fatigue={initial_state.fatigue:.1f}\n"
            f"Final risk: {r.final_risk:.2f}\n"
            f"Status: {status}\n"
            f"Total cost: {r.cost:.2f}\n"
            f"Runtime: {r.runtime*1000:.2f} ms\n"
            f"Expanded nodes: {r.nodes}\n"
            f"Final state -> attendance={r.final_state.attendance_rate:.1f}, missing={r.final_state.missing_submissions}, "
            f"quiz={r.final_state.avg_quiz_score:.1f}, lms={r.final_state.lms_activity:.1f}, study={r.final_state.study_hours_per_week:.1f}, deadline={r.final_state.days_to_deadline}"
        )
        self.plan.configure(state=tk.NORMAL)
        self.plan.delete("1.0", tk.END)
        if r.actions:
            for i, a in enumerate(r.actions, 1):
                self.plan.insert(tk.END, f"{i}. {a}\n")
        else:
            self.plan.insert(tk.END, "No plan generated.\n")
        if not r.found:
            self.plan.insert(tk.END, "\nGoal not fully reached within constraints; showing best found path.\n")
        self.plan.configure(state=tk.DISABLED)

    def _show_table(self, rows: List[Result]):
        for i in self.table.get_children():
            self.table.delete(i)
        for r in rows:
            self.table.insert(
                "",
                tk.END,
                values=(
                    r.method,
                    "Yes" if r.found else "No",
                    f"{r.cost:.2f}",
                    f"{r.runtime*1000:.2f}",
                    r.nodes,
                    f"{r.final_risk:.2f}",
                    len(r.actions),
                ),
            )

    def _show_plots(self, before: float, r: Result):
        if not HAS_MPL:
            return
        self.ax_risk.clear()
        self.ax_actions.clear()
        self.ax_risk.set_title("Risk Before vs After")
        self.ax_risk.bar(["Before", "After"], [before, r.final_risk], color=["#d65f5f", "#5f9ed6"])
        self.ax_risk.set_ylim(0, 100)
        cnt: Dict[str, int] = {}
        for a in r.actions:
            cnt[a] = cnt.get(a, 0) + 1
        if cnt:
            self.ax_actions.bar(list(cnt.keys()), list(cnt.values()), color="#83b66f")
            self.ax_actions.tick_params(axis="x", rotation=40)
        else:
            self.ax_actions.bar(["N/A"], [0], color="#83b66f")
        self.ax_actions.set_title("Action Counts")
        self.fig.tight_layout()
        self.canvas.draw()

    def run_whatif(self, show_warnings: bool = True, show_done: bool = True):
        try:
            base = int(self.deadline_shift.get())
            study_limit = float(self.study_limit.get())
            tutor_ok = bool(self.tutor_ok.get())
            cases = [
                ("Base constraints", study_limit, base, tutor_ok),
                ("Study limit 1h/day", 1.0, base, tutor_ok),
                ("Deadline 3 days closer", study_limit, base - 3, tutor_ok),
                ("Tutor unavailable", study_limit, base, False),
            ]
        except Exception as e:
            if show_warnings:
                messagebox.showerror("Input error", str(e))
            else:
                for i in self.wtable.get_children():
                    self.wtable.delete(i)
            return
        for i in self.wtable.get_children():
            self.wtable.delete(i)
        for name, study, shift, tutor in cases:
            s0 = self._state_from_source(shift=shift, show_warnings=show_warnings)
            if s0 is None:
                if not show_warnings:
                    for i in self.wtable.get_children():
                        self.wtable.delete(i)
                return
            p = Planner(s0, self._constraints(s0.days_to_deadline, study=study, tutor=tutor))
            out = p.search("astar")
            self.wtable.insert(
                "",
                tk.END,
                values=(name, "Yes" if out.found else "No", f"{out.cost:.2f}", f"{out.final_risk:.2f}", out.nodes, f"{out.runtime*1000:.2f}"),
            )
        if show_done:
            messagebox.showinfo("What-if", "Ran 3 what-if cases plus base case.")


def self_test(csv_path: Path, row: int = 0):
    data = parse_csv(csv_path)
    r = data[row]
    s0 = State(
        attendance_rate=r["attendance_rate"],
        missing_submissions=r["missing_submissions"],
        avg_quiz_score=r["avg_quiz_score"],
        lms_activity=r["lms_activity"],
        study_hours_per_week=r["study_hours_per_week"],
        days_to_deadline=r["days_to_deadline"],
        fatigue=estimate_initial_fatigue(r),
    )
    p = Planner(s0, Constraints(initial_days=s0.days_to_deadline))
    print(f"Self-test student: {r['student_id']}")
    print(f"Initial risk score: {p.risk(s0):.2f}")
    for m in ("astar", "ucs", "greedy"):
        o = p.search(m)
        print(
            f"{o.method:>6} | found={o.found!s:<5} | cost={o.cost:7.2f} | risk={o.final_risk:6.2f} | "
            f"nodes={o.nodes:4d} | runtime_ms={o.runtime*1000:8.2f} | plan_len={len(o.actions):2d}"
        )
        if o.actions:
            print("  actions:", " -> ".join(o.actions))


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Student At-Risk Recovery Planner")
    p.add_argument("--self-test", action="store_true", help="Run CLI test instead of GUI")
    p.add_argument("--csv", default="STUDEN_1.CSV", help="CSV path for self-test")
    p.add_argument("--row", type=int, default=0, help="Row index for self-test")
    return p.parse_args()


def main():
    a = args()
    if a.self_test:
        self_test(Path(a.csv), a.row)
    else:
        app = App()
        app.mainloop()


if __name__ == "__main__":
    main()

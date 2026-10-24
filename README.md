# Student At-Risk Recovery Planner (A* Search)

This project implements the CSAI 350 team assignment as a Python desktop dashboard.

## Included Features
- CSV loading for student context fields:
  - `attendance_rate`
  - `missing_submissions`
  - `avg_quiz_score`
  - `lms_activity`
  - `study_hours_per_week`
  - `days_to_deadline`
- Scenario generation (synthetic student records)
- Dual input mode:
  - Use selected student values from CSV
  - Or enter custom scenario values manually (attendance, missing, quiz, LMS, study, deadline, fatigue)
- A* planning engine from scratch:
  - Open set (priority queue)
  - Closed/best-known state tracking
  - `g(n)`, `h(n)`, `f(n)=g+h`
- State/action model with 6 actions:
  - `Attend_Class`
  - `Study_1_Hour`
  - `Submit_Assignment`
  - `Practice_Quiz`
  - `Meet_Tutor`
  - `Rest`
- Goal condition:
  - Student reaches Not At-Risk threshold and clears missing submissions
- Dashboard outputs:
  - Plan steps
  - Total cost
  - Final risk/status
  - Runtime
  - Expanded nodes
- Plots:
  - Risk before vs after
  - Action counts
- Baseline comparison table:
  - A* vs Uniform Cost Search vs Greedy best-first
- What-if simulation (3 required cases + base):
  - Study limit reduced to 1 hour/day
  - Deadline moved 3 days closer
  - Tutor unavailable
- Tunable planning settings:
  - Risk threshold goal
  - Deadline cost weight
  - Fatigue cost weight

## Run
1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Launch the desktop dashboard:

```bash
python student_recovery_dashboard.py
```

3. In the app:
- `Input source = Custom` uses manual fields (empty by default on launch).
- `Input source = CSV` uses the selected student row.
- `Copy Student -> Custom` copies a row into editable custom fields.

3. Optional CLI self-test:

```bash
python student_recovery_dashboard.py --self-test --csv STUDEN_1.CSV --row 0
```

## Notes
- If `STUDEN_1.CSV` is in the same folder, the app auto-loads it on startup.
- If matplotlib is not installed, the GUI still runs but chart panels are disabled.

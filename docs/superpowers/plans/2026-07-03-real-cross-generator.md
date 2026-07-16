# 跨生成器真实闸门 实施计划(2026-07-03)

**Goal:** 按 2026-07-02 组会汇报的下一步优先级:接入第二个真实生成器(Kandinsky 2.2 inpaint,异族 unCLIP 先验)解锁 gate3 跨生成器判定与 gate2 cross_model;gate1 增强度桶边界敏感性;GPU 复跑出 2026-07-03 真实结果。

**关键设计:**
- `diffusers_gen`:生成器按 **name→model_id 映射**(不再全部落到 SD1.5);全部管线改 `enable_model_cpu_offload()`(3 个模型 >8GB 显存,常驻会 OOM;offload 峰值≈最大单模块 2.6GB,安全);safety_checker 参数仅对 SD 系模型传。
- `registry`:real 实例缓存 key 带 name,不同 name 不同实例;meta 的 generator_name/family 如实。
- `configs/generators.real.yaml`:img2img 仅 SD1.5(保持 gate1 强度网格与 07-02 可比);inpainters = SD1.5-inpaint + kandinsky-inpaint。
- `configs/probe.real.yaml`:generators_config 指向 real 清单;`holdout_generators: [kandinsky-inpaint]` → gate2 出现 test_b split,gate3 heldout 非空。
- `gate1`:metrics 增 `bucket_sensitivity`(边界 {0.30/0.60, 0.35/0.65, 0.40/0.70} 下的 reg-bucket BA)。
- 规模:n_base=50 → gate1 250(SD)+ gate2 450(50 img2img + 200 SD-inpaint + 200 Kandinsky-inpaint);Kandinsky 权重首跑下载 ~7GB。
- **不做**(记入待办):真实 gate4 完整 pipeline split(需 txt2img 真实后端);底图场景多样性扩展。

**执行顺序:** Task A(多生成器,TDD)→ Task B(gate1 敏感性,TDD)→ CPU 全绿提交 → Task C(GPU 复跑 + `docs/real_gate_results_2026-07-03.md` + 推送)。判定如实报告,跨生成器崩了也照写(那正是第二篇泛化的动机)。

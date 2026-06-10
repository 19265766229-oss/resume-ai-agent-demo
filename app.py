import re
import json
import html
from datetime import datetime

import pandas as pd
import streamlit as st

try:
    import fitz  # PyMuPDF，用于读取 PDF
except Exception:
    fitz = None


# =========================================================
# 0. 页面配置
# =========================================================

st.set_page_config(
    page_title="AI 求职智能匹配智能体",
    page_icon="🎯",
    layout="wide"
)


# =========================================================
# 1. 文本读取与清洗
# =========================================================

def clean_text(text) -> str:
    """清洗简历文本。"""
    if text is None:
        return ""

    try:
        if pd.isna(text):
            return ""
    except Exception:
        pass

    text = str(text)
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def extract_text_from_pdf(uploaded_file) -> str:
    """从 PDF 中提取文本。"""
    if fitz is None:
        return ""

    try:
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        texts = []
        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                texts.append(page_text)

        return clean_text("\n".join(texts))

    except Exception:
        return ""


def read_uploaded_resume(uploaded_file) -> str:
    """读取上传的 PDF / TXT / MD 简历。"""
    if uploaded_file is None:
        return ""

    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)

    if filename.endswith(".txt") or filename.endswith(".md"):
        try:
            return clean_text(uploaded_file.read().decode("utf-8", errors="ignore"))
        except Exception:
            return ""

    return ""


# =========================================================
# 2. 技能词库
# =========================================================

SKILL_KEYWORDS = {
    "AI/大模型": [
        "大模型", "LLM", "AIGC", "生成式AI", "生成式 AI", "LangChain",
        "Agent", "智能体", "RAG", "检索增强生成", "向量检索", "Embedding",
        "Prompt", "提示词", "思维链", "CoT", "多轮对话", "工具调用",
        "Function Calling", "多模态", "图文理解", "知识库"
    ],
    "机器学习/深度学习": [
        "机器学习", "深度学习", "PyTorch", "TensorFlow", "Scikit-learn",
        "sklearn", "NLP", "自然语言处理", "CV", "计算机视觉", "推荐系统",
        "分类模型", "回归模型", "神经网络"
    ],
    "编程语言": [
        "Python", "Java", "C++", "C语言", "C、", "C，", "JavaScript",
        "TypeScript", "Go", "MATLAB", "SQL"
    ],
    "后端/工程开发": [
        "后端", "接口", "API", "Flask", "Django", "Spring", "Spring Boot",
        "FastAPI", "Linux", "Git", "Docker", "Kubernetes", "MySQL", "Redis",
        "MongoDB", "数据库", "系统开发", "软件开发"
    ],
    "数据分析": [
        "数据分析", "数据清洗", "数据可视化", "Pandas", "NumPy", "Excel",
        "Tableau", "PowerBI", "统计分析", "指标体系", "看板", "A/B测试"
    ],
    "计算机基础/安全": [
        "数据结构", "算法", "操作系统", "计算机网络", "网络空间安全",
        "信息安全", "Web安全", "密码学", "渗透测试", "漏洞分析"
    ],
    "产品/设计": [
        "需求分析", "用户调研", "竞品分析", "原型设计", "PRD", "Axure",
        "Figma", "UI", "UX", "产品设计", "交互设计"
    ],
    "运营/市场/销售": [
        "用户运营", "内容运营", "活动策划", "社群运营", "新媒体运营",
        "销售", "客户沟通", "商务拓展", "市场调研", "渠道"
    ],
    "通用能力": [
        "团队协作", "沟通", "组织协调", "统筹管理", "项目管理",
        "问题解决", "学习能力", "负责人", "学生会", "部长"
    ]
}


def normalize_skill_name(skill: str) -> str:
    """统一技能名。"""
    mapping = {
        "C、": "C",
        "C，": "C",
        "C语言": "C",
        "sklearn": "Scikit-learn",
        "生成式 AI": "生成式AI",
    }
    return mapping.get(skill, skill)


def extract_skills(text: str):
    """通过关键词词库识别技能。"""
    text = clean_text(text)
    lower_text = text.lower()

    found = []

    for group, keywords in SKILL_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in lower_text:
                found.append({
                    "skill": normalize_skill_name(keyword),
                    "group": group
                })

    # 去重
    seen = set()
    unique = []

    for item in found:
        key = item["skill"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)

    return unique


# =========================================================
# 3. 职业方向识别规则
# =========================================================

DIRECTION_RULES = {
    "算法类": {
        "keywords": [
            "AI 开发", "AI开发", "人工智能", "大模型", "LLM", "AIGC",
            "LangChain", "Agent", "智能体", "RAG", "向量检索", "Embedding",
            "机器学习", "深度学习", "NLP", "自然语言处理", "CV", "计算机视觉",
            "多模态", "图文理解", "知识库", "Prompt", "思维链"
        ],
        "weight": 3.2,
    },
    "技术类": {
        "keywords": [
            "Python", "Java", "C++", "C语言", "C、", "C，", "软件开发", "开发",
            "系统", "后端", "接口", "API", "数据库", "计算机网络",
            "网络空间安全", "信息安全", "Linux", "Git", "Docker", "Web安全"
        ],
        "weight": 2.6,
    },
    "数据类": {
        "keywords": [
            "数据分析", "数据清洗", "数据可视化", "SQL", "Excel", "Tableau",
            "PowerBI", "统计分析", "指标体系", "看板", "A/B测试", "Pandas", "NumPy"
        ],
        "weight": 2.1,
    },
    "产品类": {
        "keywords": [
            "产品", "需求分析", "用户调研", "竞品分析", "原型设计", "PRD",
            "Axure", "Figma", "交互", "UI", "UX"
        ],
        "weight": 1.8,
    },
    "运营类": {
        "keywords": [
            "运营", "活动策划", "社群", "新媒体", "内容运营", "用户运营",
            "组织协调", "统筹管理", "学生会", "负责人", "部长"
        ],
        "weight": 1.5,
    },
    "销售类": {
        "keywords": [
            "销售", "客户", "商务", "渠道", "市场", "客户维护", "商务拓展"
        ],
        "weight": 1.5,
    },
    "综合类": {
        "keywords": [
            "沟通", "团队协作", "学习能力", "问题解决", "自我评价"
        ],
        "weight": 1.0,
    },
}


def score_directions(text: str):
    """计算每个职业方向的匹配分数。"""
    text = clean_text(text)
    lower_text = text.lower()

    scores = {}

    for direction, rule in DIRECTION_RULES.items():
        score = 0.0
        matched = []

        for keyword in rule["keywords"]:
            if keyword.lower() in lower_text:
                score += rule["weight"]
                matched.append(normalize_skill_name(keyword))

        scores[direction] = {
            "score": round(score, 2),
            "matched": list(dict.fromkeys(matched))
        }

    # 针对中文学生 AI 简历的强信号加权
    if "ai 开发实习生" in lower_text or "ai开发实习生" in lower_text:
        scores["算法类"]["score"] += 9
        scores["技术类"]["score"] += 5

    if "多模态客服智能体" in lower_text:
        scores["算法类"]["score"] += 7
        scores["技术类"]["score"] += 3

    if "网络空间安全" in lower_text:
        scores["技术类"]["score"] += 5

    # 避免“数据结构”把计算机简历误判成数据分析方向
    if "数据结构" in text and "数据分析" not in text:
        scores["数据类"]["score"] = max(0, scores["数据类"]["score"] - 1.5)

    return scores


def predict_main_direction(text: str):
    scores = score_directions(text)
    best_direction = max(scores.items(), key=lambda x: x[1]["score"])[0]
    return best_direction, scores


def normalize_scores_for_table(scores: dict):
    """把分数缩放到 0-100，便于表格展示。"""
    raw = {k: v["score"] for k, v in scores.items()}
    max_score = max(raw.values()) if raw else 0

    if max_score <= 0:
        return {k: 0.0 for k in raw}

    return {
        k: round(v / max_score * 100, 1)
        for k, v in raw.items()
    }


# =========================================================
# 4. 岗位画像库
# =========================================================

JOB_LIBRARY = [
    {
        "title": "AI 开发实习生",
        "category": "算法类",
        "description": "参与大模型应用、智能体、RAG 知识库、多轮对话和工具调用能力开发。",
        "must_have": ["Python", "大模型", "LangChain", "RAG", "向量检索", "多轮对话"],
        "nice_have": ["FastAPI", "Prompt", "多模态", "图文理解", "Embedding"],
        "tasks": ["搭建轻量化 Agent", "构建知识库问答", "优化提示词与检索效果"],
    },
    {
        "title": "大模型应用开发实习生",
        "category": "算法类",
        "description": "围绕 LLM 应用场景进行系统设计、知识库问答、业务流程自动化和评测优化。",
        "must_have": ["Python", "大模型", "RAG", "LangChain", "Prompt"],
        "nice_have": ["向量检索", "Function Calling", "多模态", "FastAPI"],
        "tasks": ["实现 LLM 应用 Demo", "设计 RAG 流程", "评估回答质量"],
    },
    {
        "title": "后端开发实习生",
        "category": "技术类",
        "description": "参与业务系统后端接口、数据库和服务端功能开发。",
        "must_have": ["Java", "Python", "数据库", "Git"],
        "nice_have": ["Spring Boot", "MySQL", "Redis", "Linux", "Docker"],
        "tasks": ["开发后端接口", "设计数据库表", "联调前后端功能"],
    },
    {
        "title": "网络安全实习生",
        "category": "技术类",
        "description": "参与安全测试、漏洞分析、日志分析和安全工具开发。",
        "must_have": ["网络空间安全", "计算机网络", "Python", "信息安全"],
        "nice_have": ["Web安全", "Linux", "漏洞分析", "渗透测试"],
        "tasks": ["安全测试", "分析漏洞风险", "编写安全脚本"],
    },
    {
        "title": "数据分析实习生",
        "category": "数据类",
        "description": "使用数据分析方法支持业务决策、指标监控和报表分析。",
        "must_have": ["SQL", "Excel", "数据分析", "统计分析"],
        "nice_have": ["Python", "Pandas", "Tableau", "PowerBI"],
        "tasks": ["清洗业务数据", "搭建指标看板", "输出分析报告"],
    },
    {
        "title": "产品经理实习生",
        "category": "产品类",
        "description": "参与需求分析、用户调研、竞品分析和产品方案设计。",
        "must_have": ["需求分析", "用户调研", "产品", "沟通"],
        "nice_have": ["原型设计", "PRD", "Axure", "Figma"],
        "tasks": ["整理用户需求", "绘制产品原型", "跟进研发进度"],
    },
    {
        "title": "运营实习生",
        "category": "运营类",
        "description": "参与内容运营、活动策划、用户增长和社群维护。",
        "must_have": ["运营", "活动策划", "沟通", "组织协调"],
        "nice_have": ["新媒体运营", "内容运营", "用户运营", "数据分析"],
        "tasks": ["策划活动", "维护用户社群", "分析运营数据"],
    },
    {
        "title": "销售管培生",
        "category": "销售类",
        "description": "参与客户拓展、商务沟通、市场调研和销售转化。",
        "must_have": ["销售", "客户沟通", "商务拓展", "市场"],
        "nice_have": ["渠道", "客户维护", "数据分析"],
        "tasks": ["跟进客户", "整理销售线索", "完成商务沟通"],
    },
]


def get_skill_names(skills):
    return [item["skill"] for item in skills]


def evaluate_resume_expression(text: str):
    """评估简历表达完整度。"""
    text = clean_text(text)
    score = 0

    if any(k in text for k in ["项目经验", "实习", "工作经历", "项目"]):
        score += 4

    if any(k in text for k in ["负责", "搭建", "构建", "实现", "优化", "开发"]):
        score += 3

    if any(k in text for k in ["%", "提升", "降低", "准确率", "召回率", "耗时", "有效"]):
        score += 3

    return min(score, 10)


def match_job(resume_text: str, job: dict):
    """计算简历与单个岗位的匹配度。"""
    skills = get_skill_names(extract_skills(resume_text))
    skill_set = set([s.lower() for s in skills])
    lower_text = resume_text.lower()

    matched_must = []
    missing_must = []
    matched_nice = []

    for skill in job.get("must_have", []):
        if skill.lower() in skill_set or skill.lower() in lower_text:
            matched_must.append(skill)
        else:
            missing_must.append(skill)

    for skill in job.get("nice_have", []):
        if skill.lower() in skill_set or skill.lower() in lower_text:
            matched_nice.append(skill)

    must_have = job.get("must_have", [])
    nice_have = job.get("nice_have", [])

    must_score = len(matched_must) / len(must_have) * 55 if must_have else 20
    nice_score = len(matched_nice) / len(nice_have) * 20 if nice_have else 5

    main_direction, _ = predict_main_direction(resume_text)
    direction_score = 15 if main_direction == job["category"] else 6

    expression_score = evaluate_resume_expression(resume_text)

    total = must_score + nice_score + direction_score + expression_score
    total = round(min(total, 100), 1)

    return {
        "title": job["title"],
        "category": job["category"],
        "description": job["description"],
        "score": total,
        "matched_must": matched_must,
        "missing_must": missing_must,
        "matched_nice": matched_nice,
        "tasks": job.get("tasks", []),
    }


def rank_jobs(resume_text: str, top_k=5):
    results = [match_job(resume_text, job) for job in JOB_LIBRARY]
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def generate_improvement_suggestions(resume_text: str, target_job_result: dict):
    """生成简历优化建议。"""
    suggestions = []

    if target_job_result["missing_must"]:
        suggestions.append(
            "补充岗位核心技能：建议在简历中更明确体现 "
            + "、".join(target_job_result["missing_must"])
            + "，可以写在技能栏或项目经历中。"
        )

    if not any(k in resume_text for k in ["%", "提升", "降低", "准确率", "召回率", "耗时"]):
        suggestions.append(
            "增加量化结果：例如“将问答命中率提升至 xx%”“响应时间降低 xx%”“覆盖 xx 份知识文档”。"
        )

    if "项目经验" in resume_text and not any(k in resume_text for k in ["技术栈", "项目难点", "解决方案"]):
        suggestions.append(
            "优化项目表达：建议每段项目按“背景-任务-技术栈-行动-结果”组织，突出技术难点和个人贡献。"
        )

    if target_job_result["category"] in ["算法类", "技术类"]:
        suggestions.append(
            "针对技术岗位：建议单独增加“技能栈”模块，列出 Python、LangChain、RAG、向量检索、数据库、Git 等关键词，方便 HR 和 ATS 快速识别。"
        )

    if not suggestions:
        suggestions.append("当前简历与目标岗位匹配度较好，可进一步补充项目成果数据和岗位关键词。")

    return suggestions


# =========================================================
# 5. 稳定展示组件
# =========================================================

def show_result_cards(main_direction, skill_count, highest_score):
    """不用 st.metric，改用 HTML 卡片，避免前端动态模块加载失败。"""
    main_direction_safe = html.escape(str(main_direction))
    skill_count_safe = html.escape(str(skill_count))
    highest_score_safe = html.escape(str(highest_score))

    st.markdown(
        f"""
        <div style="display: flex; gap: 16px; margin: 16px 0 24px 0;">
            <div style="flex: 1; padding: 20px; border: 1px solid #e5e7eb; border-radius: 12px; background-color: #f9fafb;">
                <div style="font-size: 15px; color: #6b7280; margin-bottom: 8px;">主要方向</div>
                <div style="font-size: 30px; font-weight: 700; color: #111827;">{main_direction_safe}</div>
            </div>
            <div style="flex: 1; padding: 20px; border: 1px solid #e5e7eb; border-radius: 12px; background-color: #f9fafb;">
                <div style="font-size: 15px; color: #6b7280; margin-bottom: 8px;">识别技能数</div>
                <div style="font-size: 30px; font-weight: 700; color: #111827;">{skill_count_safe}</div>
            </div>
            <div style="flex: 1; padding: 20px; border: 1px solid #e5e7eb; border-radius: 12px; background-color: #f9fafb;">
                <div style="font-size: 15px; color: #6b7280; margin-bottom: 8px;">最高岗位匹配度</div>
                <div style="font-size: 30px; font-weight: 700; color: #111827;">{highest_score_safe}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_score_card(label, value):
    """目标岗位匹配度卡片。"""
    label_safe = html.escape(str(label))
    value_safe = html.escape(str(value))

    st.markdown(
        f"""
        <div style="padding: 20px; border: 1px solid #e5e7eb; border-radius: 12px; background-color: #f9fafb; margin-bottom: 16px;">
            <div style="font-size: 15px; color: #6b7280; margin-bottom: 8px;">{label_safe}</div>
            <div style="font-size: 30px; font-weight: 700; color: #111827;">{value_safe}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# 6. 页面主体
# =========================================================

st.title("🎯 AI 求职智能匹配智能体")
st.caption("帮助学生识别简历方向、匹配合适岗位，并给出面向目标岗位的简历优化建议。")

with st.sidebar:
    st.header("使用说明")
    st.write("1. 上传 PDF 简历，或直接粘贴简历文本。")
    st.write("2. 系统自动识别技能关键词与职业方向。")
    st.write("3. 查看推荐岗位、匹配度和简历优化建议。")
    st.divider()
    st.write("本 Demo 不依赖招聘 API 或爬虫，使用规则识别 + 岗位画像库实现。")

uploaded_file = st.file_uploader(
    "上传简历文件，支持 PDF / TXT / MD",
    type=["pdf", "txt", "md"]
)

pasted_text = st.text_area(
    "也可以直接粘贴简历文本",
    height=180,
    placeholder="请粘贴你的简历文本……"
)

resume_text = ""

if uploaded_file is not None:
    resume_text = read_uploaded_resume(uploaded_file)

if pasted_text.strip():
    resume_text = clean_text(pasted_text)

if not resume_text:
    st.info("请先上传简历或粘贴简历文本。")
    st.stop()

with st.expander("查看系统提取到的简历文本，调试用", expanded=False):
    st.write("文本长度：", len(resume_text))
    st.text_area("提取文本预览", resume_text[:3000], height=260)

skills = extract_skills(resume_text)
main_direction, direction_scores = predict_main_direction(resume_text)
direction_index_scores = normalize_scores_for_table(direction_scores)
top_jobs = rank_jobs(resume_text, top_k=5)

highest_score = f"{top_jobs[0]['score']} 分" if top_jobs else "暂无"

st.header("一、简历画像与职业方向识别")

show_result_cards(
    main_direction=main_direction,
    skill_count=len(skills),
    highest_score=highest_score
)

st.subheader("方向匹配得分")

direction_rows = []

for direction, value in direction_index_scores.items():
    matched_keywords = direction_scores[direction]["matched"]
    direction_rows.append({
        "职业方向": direction,
        "匹配指数": value,
        "命中关键词": "、".join(matched_keywords) if matched_keywords else "暂无"
    })

direction_df = pd.DataFrame(direction_rows).sort_values("匹配指数", ascending=False)

# 用 table 替代 bar_chart，避免 ArrowVegaLiteChart 前端模块加载失败
st.table(direction_df)

st.subheader("识别到的技能关键词")

if skills:
    skill_df = pd.DataFrame(skills)
    st.dataframe(skill_df, use_container_width=True)
else:
    st.warning("暂未识别到明显技能关键词。建议检查 PDF 文本是否成功提取，或在简历中增加技能栈模块。")

st.header("二、岗位智能推荐")

for i, job in enumerate(top_jobs, start=1):
    with st.container(border=True):
        st.subheader(f"Top {i}｜{job['title']}｜{job['score']} 分")
        st.write(job["description"])
        st.write("岗位方向：", job["category"])
        st.write("已匹配核心技能：", "、".join(job["matched_must"]) if job["matched_must"] else "暂无")
        st.write("待补充核心技能：", "、".join(job["missing_must"]) if job["missing_must"] else "无明显缺口")
        st.write("加分技能：", "、".join(job["matched_nice"]) if job["matched_nice"] else "暂无")

st.header("三、心仪岗位匹配度与简历优化建议")

job_titles = [job["title"] for job in JOB_LIBRARY]
selected_title = st.selectbox("选择你的心仪岗位", job_titles)

selected_job = next(job for job in JOB_LIBRARY if job["title"] == selected_title)
target_result = match_job(resume_text, selected_job)

left_col, right_col = st.columns([1, 2])

with left_col:
    show_score_card("目标岗位匹配度", f"{target_result['score']} 分")

with right_col:
    st.write("目标岗位：", target_result["title"])
    st.write("岗位说明：", target_result["description"])
    st.write("岗位方向：", target_result["category"])

st.write("已匹配核心技能：", "、".join(target_result["matched_must"]) if target_result["matched_must"] else "暂无")
st.write("待补充核心技能：", "、".join(target_result["missing_must"]) if target_result["missing_must"] else "无明显缺口")
st.write("已匹配加分技能：", "、".join(target_result["matched_nice"]) if target_result["matched_nice"] else "暂无")

suggestions = generate_improvement_suggestions(resume_text, target_result)

st.subheader("简历优化建议")

for suggestion in suggestions:
    st.markdown(f"- {suggestion}")

report = {
    "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "主要方向": main_direction,
    "识别技能": [item["skill"] for item in skills],
    "方向匹配得分": direction_rows,
    "推荐岗位Top5": top_jobs,
    "目标岗位": target_result,
    "优化建议": suggestions,
}

st.download_button(
    "下载匹配报告 JSON",
    data=json.dumps(report, ensure_ascii=False, indent=2),
    file_name="resume_match_report.json",
    mime="application/json"
)
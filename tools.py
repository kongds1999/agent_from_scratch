from agents.decorators import function_tool
from typing import Annotated, Dict, Optional
import os, json, subprocess, glob as _glob


# Global registry to store file contents before modification to allow 'undo' operations
# Key: File path (str) | Value: Content (str) or None if file was newly created
SNAPSHOTS: Dict[str, Optional[str]] = {}

# 1. 危险命令黑名单与安全校验
ALWAYS_BLOCKED_COMMANDS = ["rm", "sudo", "shutdown", "reboot", "dd", "mkfs", "kill", "killall", "> /dev/", ":(){ :|:& };:"]

@function_tool
def run_bash(command: Annotated[str, "The bash command to run"]) -> str:
    """Executes a shell command synchronously in the current working directory."""
    # UI: 黄色高亮打印工具调用，参考 01_agent_loop.py / core.py 中的 dispatch_tools
    print(f"\033[33m[bash] {command[:80]}...\033[0m")

    if any(blocked in command for blocked in ALWAYS_BLOCKED_COMMANDS):
        output = "Error: dangerous command blocked by safety policy"
        print(output)
        return output
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = (result.stdout + result.stderr).strip()
        res = output[:50000] if output else "(no output)"
        # 打印 run_bash 执行的原始输出日志（截断前300字符），参考 core.py
        print(str(res)[:300])
        return res
    except subprocess.TimeoutExpired:
        output = "Error: timeout (120s)"
        print(output)
        return output
    except Exception as e:
        output = f"Error: {e}"
        print(output)
        return output
    

@function_tool
def read_file(
    path: Annotated[str, "Path to the file"], 
    start_line: Annotated[int, "The starting line number (1-indexed)"] = None, 
    end_line: Annotated[int, "The ending line number (1-indexed)"] = None
) -> str:
    """Reads a file from disk with optional line-range slicing and line numbering."""
    try:
        with open(path, 'r', encoding='utf-8', errors="replace") as f:
            lines = f.readlines()
        
        # Convert 1-based human/AI indexing to 0-based Python indexing
        start_index = (start_line or 1) - 1
        # Set end index to requested line or default to end of file
        end_index = end_line or len(lines)
        
        # Build a string where every line is prefixed by its line number
        numbered_lines = "".join(
            f"{start_index + 1 + i:4d}\t{line}" 
            for i, line in enumerate(lines[start_index:end_index])
        )
        # Return formatted text, capped at 50k chars
        return numbered_lines[:50000] or "(empty file)"
    except FileNotFoundError:
        # Explicit error for missing files
        return f"Error: file not found: {path}"
    except Exception as e:
        # Generic error handling for permission issues, etc.
        return f"Error reading {path}: {e}"

@function_tool
def write_file(
    path: Annotated[str, "Destination path to write to"],
    content: Annotated[str, "Content to write to the file"]
) -> str:
    """Writes content to a file and stores a snapshot for potential restoration."""
    try:
        # Check if file exists to determine if we update or create
        if os.path.exists(path):
            # Read and store current content for 'revert' functionality
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                SNAPSHOTS[path] = f.read()
            action = "updated"
        else:
            # Mark as None in snapshots so 'revert' knows to delete the file
            SNAPSHOTS[path] = None
            action = "created"
        
        # Ensure the directory structure exists before writing the file
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Perform the actual file write
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"{action}: {path} (snapshot saved — use revert to undo)"
    except Exception as e:
        # Capture and return write-related exceptions
        return f"Error writing {path}: {e}"
    
@function_tool
def run_grep(
    pattern: Annotated[str, "The regex pattern to search for"],
    path: Annotated[str, "Path to the file to search in"],
    recursive: Annotated[bool, "Whether to search recursively in directories"] = False
) -> str:
    """Performs a regex search within files using system grep or Windows findstr."""
    try:
        # Determine recursion flag for Unix grep
        flags = ["-r"] if recursive else []
        # Execute grep with line numbers (-n)
        result = subprocess.run(
            ["grep", "-n", *flags, pattern, path],
            capture_output=True, text=True, timeout=30
        )
        # Return results truncated to 10k chars to keep context lean
        return ((result.stdout + result.stderr).strip() or "(no matches)")[:10000]
    except FileNotFoundError:
        # Fallback mechanism for Windows environments without grep installed
        try:
            # Construct findstr command for common code file extensions
            command = f'findstr /S /N "{pattern}" "{path}\\*.py" "{path}\\*.js" "{path}\\*.md"'
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=30
            )
            return ((result.stdout + result.stderr).strip() or "(no matches)")[:10000]
        except Exception as e:
            return f"Error: grep/findstr failed: {e}"
    except subprocess.TimeoutExpired:
        return "Error: grep timeout"
    except Exception as e:
        return f"Error: {e}"
    
@function_tool
def run_glob(
    pattern: Annotated[str, "The glob pattern to match files"]
) -> str:
    """Locates files matching a specific glob pattern."""
    # Perform recursive glob search using the standard library
    matches = _glob.glob(pattern, recursive=True)
    if not matches:
        return "(no matches)"
    # Sort for consistency and limit count to prevent massive context inflation
    return "\n".join(sorted(matches)[:200])


@function_tool
def revert_file(
    path: Annotated[str, "Path to the file to revert"]
) -> str:
    """Reverts a file to its previous state using the stored snapshot."""
    # Check if a snapshot exists for this path
    if path not in SNAPSHOTS:
        return f"Error: no snapshot for {path}"
    
    # Retrieve and remove the snapshot from memory
    original_content = SNAPSHOTS.pop(path)
    
    if original_content is None:
        # If original_content was None, the file didn't exist before 'write'
        try:
            os.remove(path) # Revert by deleting the new file
            return f"reverted: deleted {path} (it was a new file)"
        except Exception as e:
            return f"Error deleting {path}: {e}"
    else:
        # If original_content existed, write it back to the file
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(original_content)
            return f"reverted: {path}"
        except Exception as e:
            return f"Error reverting {path}: {e}"

TODO_FILE: str = ".agent_todo.json"
@function_tool
def run_todo_write(tasks: Annotated[list[str], "tasks (List[str]): A list of strings describing each step of the plan."]) -> str:
    """
    Initializes a new task plan and saves it to the persistent JSON store.
    
        This function wipes any existing plan and starts fresh with the 
        provided list of task descriptions.
    """
    # Transform raw strings into a list of structured dictionaries with metadata
    data = [
        {"id": i, "task": t, "status": "pending"} 
        for i, t in enumerate(tasks)
    ]
    
    # Context manager ensures the file is closed properly after writing
    with open(TODO_FILE, "w", encoding="utf-8") as f:
        # Write the JSON data with indentation for human readability if opened manually
        json.dump(data, f, indent=2)
    
    # Construct a formatted preview of the plan for the agent's context
    lines = "\n".join(f"  [{i}] {t}" for i, t in enumerate(tasks))
    return f"Plan written ({len(tasks)} tasks):\n{lines}"

@function_tool
def run_todo_read() -> str:
    """
    Reads and returns the current state of the task plan.

    Used by the agent to remind itself of the next step or overall progress.

    Returns:
        str: A formatted string representing the todo list or a 'not found' message.
    """
    try:
        # Open and load the existing JSON plan
        with open(TODO_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Format each task with its ID and status (padded to 12 chars for alignment)
        return "\n".join(
            f"[{t['id']}] [{t['status']:12s}] {t['task']}" for t in data
        )
    except FileNotFoundError:
        # Graceful handling if the agent tries to read before writing
        return "(no todo list found - please use todo_write first)"
    except Exception as e:
        # General error fallback
        return f"Error reading todo list: {e}"
    
@function_tool
def run_todo_update(
    index: Annotated[int, "The numeric ID (0-based) of the task to modify"], 
    status: Annotated[str, "The new status string (e.g., 'in_progress', 'done')"]
) -> str:
    """
    Updates the completion status of a specific task within the plan.

    Returns:
        str: A confirmation message or an error description.
    """
    try:
        # Read the current state to perform an in-memory update
        with open(TODO_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Verify the index exists in the list to prevent out-of-bounds errors
        if 0 <= index < len(data):
            # Update the status value for the specific dictionary entry
            data[index]["status"] = status
            
            # Persist the modified list back to disk
            with open(TODO_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            
            return f"Updated task {index} status to: {status}"
        
        # Error if the provided index is invalid
        return f"Error: Task index {index} is out of range."
        
    except FileNotFoundError:
        return "Error: No todo list found to update."
    except Exception as e:
        return f"Error during todo_update: {e}"


####################
# Skill Management
####################
from pathlib import Path


def _resolve_skills_dir() -> Path:
    """定位 skills 目录，按优先级依次尝试：

    1. 环境变量 ``SKILLS_DIR``（显式覆盖，便于部署到别处）
    2. 从当前文件向上查找第一个存在的 ``<parent>/skills`` 目录
    3. 回退到当前文件上级目录下的 ``skills/``

    这样既兼容 ``src/skills/``，也兼容仓库根目录的 ``skills/``。
    """
    override = os.environ.get("SKILLS_DIR")
    if override:
        return Path(override).expanduser().resolve()

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "skills"
        if candidate.is_dir():
            return candidate

    return here.parent.parent / "skills"


SKILLS_DIR: Path = _resolve_skills_dir()


def discover_skills() -> Dict[str, str]:
    """
    Scans the skills directory and extracts metadata from SKILL.md files.

    It parses the first non-empty line of text (ignoring YAML frontmatter) 
    to use as a brief description for the agent's index.

    Returns:
        Dict[str, str]: A dictionary mapping {skill_name: short_description}.
    """
    skills: Dict[str, str] = {}
    
    # Ensure the skills directory actually exists to avoid iteration errors
    if not SKILLS_DIR.exists():
        return skills

    # Iterate through subdirectories in alphabetical order
    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        skill_md = skill_dir / "SKILL.md"
        
        # We only consider directories that contain a SKILL.md file
        if skill_dir.is_dir() and skill_md.exists():
            try:
                # Parse YAML frontmatter (between leading `---` and closing `---`)
                # to extract the canonical `name` and `description` fields.
                lines = skill_md.read_text(encoding="utf-8").splitlines()
                name: str = skill_dir.name  # fallback to folder name
                description: str = "No description available."
                in_frontmatter = False

                for line in lines:
                    stripped = line.strip()
                    if stripped == "---":
                        if not in_frontmatter:
                            # entering frontmatter
                            in_frontmatter = True
                            continue
                        else:
                            # leaving frontmatter — we have what we need
                            break

                    if in_frontmatter:
                        # Match `key: value` lines only for the two fields we care about
                        if stripped.startswith("name:"):
                            name = stripped[len("name:"):].strip()
                        elif stripped.startswith("description:"):
                            description = stripped[len("description:"):].strip()

                # Cap description length for prompt brevity
                skills[name] = description[:100]
            except Exception as e:
                # Log error and continue to the next skill
                skills[skill_dir.name] = f"Error reading metadata: {e}"
                
    return skills

@function_tool(name_override="list_skills")
def run_list_skills() -> str:
    """
    Formats the list of discovered skills for the agent's tool output.

    Returns:
        str: A formatted string list of available skills.
    """
    skills = discover_skills()
    if not skills:
        return "(no skills found in skills/ directory)"
    
    # Format as a bulleted list for the LLM's consumption
    return "\n".join(f"  - {name}: {desc}" for name, desc in skills.items())


@function_tool(name_override="load_skill")
def run_load_skill(
    name: Annotated[str, "The exact name of the skill folder to load (see list_skills)."]
) -> str:
    """
    Loads the full content of a specific skill file into the context.

    Args:
        name (str): The folder name of the skill to load.

    Returns:
        str: The full text content of the skill, or an error message.
    """
    # Sanitize and build the path to the skill file
    skill_path = SKILLS_DIR / name / "SKILL.md"
    
    # Check for existence and potential directory traversal attempts
    if not skill_path.exists():
        return f"Error: skill '{name}' not found. Use list_skills to see valid names."
    
    try:
        # Load the full documentation
        content = skill_path.read_text(encoding="utf-8")
        return f"=== SKILL: {name} ===\n\n{content}\n\n=== END SKILL ==="
    except Exception as e:
        return f"Error loading skill '{name}': {e}"


AGENT_TOOLS = [run_bash, read_file, write_file, run_grep, run_glob, revert_file, run_todo_write, run_todo_read, run_todo_update]

# "Meta-tooling" 工具：按需发现 / 加载 skill（见 05_skill_loading.py）
SKILL_TOOLS = [run_list_skills, run_load_skill]
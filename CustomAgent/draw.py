import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Rectangle, Circle
import numpy as np

def create_accurate_compliance_workflow():
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 14))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    start_color = '#4CAF50' 
    process_color = '#2196F3'
    decision_color = '#FF9800'
    streamlit_color = '#FF6B6B'
    rag_color = '#9C27B0'
    section_color = '#FFFF00'
    langgraph_color = '#00BCD4'
    
    ax.text(9, 13.5, 'Medical Compliance Checker - Actual Code Flow', 
            fontsize=18, fontweight='bold', ha='center')
    
    langgraph_box = Rectangle((0.5, 11), 5, 2, 
                              facecolor=langgraph_color, alpha=0.2, 
                              edgecolor=langgraph_color, linewidth=2)
    ax.add_patch(langgraph_box)
    ax.text(3, 12.7, 'LangGraph (Simple)', fontsize=12, fontweight='bold', ha='center')
    
    classifier_box = FancyBboxPatch((1, 11.8), 1.8, 0.6, 
                                    boxstyle="round,pad=0.05", 
                                    facecolor=langgraph_color, 
                                    edgecolor='black', linewidth=1)
    ax.add_patch(classifier_box)
    ax.text(1.9, 12.1, 'classifier_node()', fontsize=8, fontweight='bold', 
            ha='center', va='center', color='white')
    
    reasoning_box = FancyBboxPatch((3.2, 11.8), 1.8, 0.6, 
                                   boxstyle="round,pad=0.05", 
                                   facecolor=langgraph_color, 
                                   edgecolor='black', linewidth=1)
    ax.add_patch(reasoning_box)
    ax.text(4.1, 12.1, 'reasoning_node()', fontsize=8, fontweight='bold', 
            ha='center', va='center', color='white')
    
    ax.text(12, 12.7, 'Streamlit State Management Flow', fontsize=12, fontweight='bold', ha='center')
    
    start_box = FancyBboxPatch((8, 11.5), 2, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor=start_color, 
                               edgecolor='black', linewidth=2)
    ax.add_patch(start_box)
    ax.text(9, 11.9, 'START\nUser Input', fontsize=10, fontweight='bold', ha='center', va='center')
    
    step1_section = Rectangle((7, 10.2), 4, 1, 
                              facecolor=section_color, alpha=0.3, 
                              edgecolor='black', linewidth=1)
    ax.add_patch(step1_section)
    ax.text(9, 10.9, 'INITIAL COMPLIANCE ASSESSMENT', fontsize=10, fontweight='bold', ha='center')
    
    initial_check = FancyBboxPatch((7.5, 10.4), 3, 0.6, 
                                   boxstyle="round,pad=0.05", 
                                   facecolor=process_color, 
                                   edgecolor='black', linewidth=1)
    ax.add_patch(initial_check)
    ax.text(9, 10.7, 'graph.invoke(classifier_node)\nSelected Regulation Only', fontsize=8, fontweight='bold', 
            ha='center', va='center', color='white')
    
    decision1_box = Rectangle((7, 9), 4, 1, 
                              facecolor=streamlit_color, alpha=0.3, 
                              edgecolor='black', linewidth=1)
    ax.add_patch(decision1_box)
    ax.text(9, 9.7, 'USER DECISION', fontsize=10, fontweight='bold', ha='center')
    
    button1_yes = FancyBboxPatch((7.2, 9.2), 1.5, 0.4, 
                                 boxstyle="round,pad=0.05", 
                                 facecolor=decision_color, 
                                 edgecolor='black', linewidth=1)
    ax.add_patch(button1_yes)
    ax.text(7.95, 9.4, 'Check Other\nRegulations', fontsize=7, fontweight='bold', ha='center', va='center')
    
    button1_no = FancyBboxPatch((9.3, 9.2), 1.5, 0.4, 
                                boxstyle="round,pad=0.05", 
                                facecolor=decision_color, 
                                edgecolor='black', linewidth=1)
    ax.add_patch(button1_no)
    ax.text(10.05, 9.4, 'Skip Other\nRegulations', fontsize=7, fontweight='bold', ha='center', va='center')
    
    step2_section = Rectangle((7, 7.5), 4, 1.2, 
                              facecolor=section_color, alpha=0.3, 
                              edgecolor='black', linewidth=1)
    ax.add_patch(step2_section)
    ax.text(9, 8.4, 'MULTI-REGULATION COMPLIANCE', fontsize=10, fontweight='bold', ha='center')
    
    regs = ['FDA', 'EMA', 'HSA']
    for i, reg in enumerate(regs):
        reg_box = FancyBboxPatch((7.2 + i*1.2, 7.7), 1, 0.5, 
                                 boxstyle="round,pad=0.05", 
                                 facecolor=process_color, 
                                 edgecolor='black', linewidth=1)
        ax.add_patch(reg_box)
        ax.text(7.7 + i*1.2, 7.95, f'{reg}\nCheck', fontsize=7, fontweight='bold', 
                ha='center', va='center', color='white')
    
    decision2_box = Rectangle((7, 6.2), 4, 1, 
                              facecolor=streamlit_color, alpha=0.3, 
                              edgecolor='black', linewidth=1)
    ax.add_patch(decision2_box)
    ax.text(9, 6.9, 'USER DECISION', fontsize=10, fontweight='bold', ha='center')
    
    button2_yes = FancyBboxPatch((7.2, 6.4), 1.5, 0.4, 
                                 boxstyle="round,pad=0.05", 
                                 facecolor=decision_color, 
                                 edgecolor='black', linewidth=1)
    ax.add_patch(button2_yes)
    ax.text(7.95, 6.6, 'Provide\nExplanation', fontsize=7, fontweight='bold', ha='center', va='center')
    
    button2_no = FancyBboxPatch((9.3, 6.4), 1.5, 0.4, 
                                boxstyle="round,pad=0.05", 
                                facecolor=decision_color, 
                                edgecolor='black', linewidth=1)
    ax.add_patch(button2_no)
    ax.text(10.05, 6.6, 'Skip\nExplanation', fontsize=7, fontweight='bold', ha='center', va='center')
    
    step3_section = Rectangle((7, 4.7), 4, 1.2, 
                              facecolor=section_color, alpha=0.3, 
                              edgecolor='black', linewidth=1)
    ax.add_patch(step3_section)
    ax.text(9, 5.6, 'DETAILED VIOLATION EXPLANATIONS', fontsize=10, fontweight='bold', ha='center')
    
    explanation_box = FancyBboxPatch((7.5, 4.9), 3, 0.6, 
                                     boxstyle="round,pad=0.05", 
                                     facecolor=process_color, 
                                     edgecolor='black', linewidth=1)
    ax.add_patch(explanation_box)
    ax.text(9, 5.2, 'Direct reasoning_node() calls\nfor each non-compliant regulation', fontsize=8, fontweight='bold', 
            ha='center', va='center', color='white')
    
    decision3_box = Rectangle((7, 3.4), 4, 1, 
                              facecolor=streamlit_color, alpha=0.3, 
                              edgecolor='black', linewidth=1)
    ax.add_patch(decision3_box)
    ax.text(9, 4.1, 'USER DECISION', fontsize=10, fontweight='bold', ha='center')
    
    button3_yes = FancyBboxPatch((7.2, 3.6), 1.5, 0.4, 
                                 boxstyle="round,pad=0.05", 
                                 facecolor=decision_color, 
                                 edgecolor='black', linewidth=1)
    ax.add_patch(button3_yes)
    ax.text(7.95, 3.8, 'Deep Document\nSearch', fontsize=7, fontweight='bold', ha='center', va='center')
    
    button3_no = FancyBboxPatch((9.3, 3.6), 1.5, 0.4, 
                                boxstyle="round,pad=0.05", 
                                facecolor=decision_color, 
                                edgecolor='black', linewidth=1)
    ax.add_patch(button3_no)
    ax.text(10.05, 3.8, 'Skip Document\nSearch', fontsize=7, fontweight='bold', ha='center', va='center')
    
    rag_section = Rectangle((12, 2), 5.5, 3, 
                            facecolor=rag_color, alpha=0.2, 
                            edgecolor=rag_color, linewidth=2)
    ax.add_patch(rag_section)
    ax.text(14.75, 4.7, 'RAG Document Search', fontsize=12, fontweight='bold', ha='center')
    
    rag_components = [
        ('extract_things_to_find()', 12.5, 4.2, 'Identify search terms'),
        ('advanced_rag_query()', 12.5, 3.6, 'Query ChromaDB'),
        ('ChromaDB Vector Store', 15.5, 4.2, 'Document database'),
        ('LLM Final Analysis', 15.5, 3.6, 'Generate response'),
        ('Comprehensive Report', 14, 2.8, 'Final output')
    ]
    
    for comp, x, y, desc in rag_components:
        if 'ChromaDB' in comp:
            color = '#4DB6AC'
        elif 'LLM' in comp:
            color = '#FF7043'
        else:
            color = rag_color
            
        comp_box = FancyBboxPatch((x-0.7, y-0.25), 1.4, 0.5, 
                                  boxstyle="round,pad=0.05", 
                                  facecolor=color, 
                                  edgecolor='black', linewidth=1)
        ax.add_patch(comp_box)
        ax.text(x, y, comp.replace('()', ''), fontsize=7, fontweight='bold', 
                ha='center', va='center', color='white')
    
    end_box = FancyBboxPatch((8, 1.2), 2, 0.8, 
                             boxstyle="round,pad=0.1", 
                             facecolor='#F44336', 
                             edgecolor='black', linewidth=2)
    ax.add_patch(end_box)
    ax.text(9, 1.6, 'END\nResults', fontsize=10, fontweight='bold', 
            ha='center', va='center', color='white')
    
    main_flow_arrows = [
        ((9, 11.5), (9, 11.2)),
        ((9, 10.2), (9, 10)),
        ((9, 9), (9, 8.7)),
        ((9, 7.5), (9, 7.2)),
        ((9, 6.2), (9, 5.9)),
        ((9, 4.7), (9, 4.4)),
        ((9, 3.4), (9, 2.0)),
    ]
    
    for start, end in main_flow_arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                              arrowstyle="->", shrinkA=3, shrinkB=3,
                              mutation_scale=15, fc="black", lw=2)
        ax.add_patch(arrow)
    
    rag_internal_arrows = [
        ((13.2, 4.2), (14.8, 4.2)),
        ((13.2, 3.6), (14.8, 3.6)),
        ((14.7, 3.0), (14, 3.0)),
    ]
    
    for start, end in rag_internal_arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                              arrowstyle="->", shrinkA=3, shrinkB=3,
                              mutation_scale=15, fc="purple", lw=2)
        ax.add_patch(arrow)
    

    numbered_connections = [
        ("A", 6.8, 3.5, "Deep Document Search button"),
        ("B", 12.0, 4.5, "RAG Entry Point"),
        ("C", 14.5, 2.2, "RAG Final Report"),
        ("D", 11.2, 9.0, "Skip Other Regulations button"),
        ("E", 11.2, 6.2, "Skip Explanation button"), 
        ("F", 11.2, 3.2, "Skip Document Search button"),
        ("G", 7.8, 1.5, "END Point")
    ]
    
    for num, x, y, desc in numbered_connections:
        circle = Circle((x, y), 0.25, facecolor='red', edgecolor='white', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, num, fontsize=12, fontweight='bold', ha='center', va='center', color='white')
    
    ax.text(0.5, 6.0, 'Numbered Connections (Long Paths):', fontsize=12, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    connection_list = [
        ("A → B:", "Deep Document Search to RAG", 'orange'),
        ("B → C:", "RAG Processing Complete", 'purple'), 
        ("C → G:", "RAG Results to END", 'purple'),
        ("D → G:", "Skip Other Regulations to END", 'gray'),
        ("E → G:", "Skip Explanation to END", 'gray'),
        ("F → G:", "Skip Document Search to END", 'gray')
    ]
    
    for i, (arrow, desc, color) in enumerate(connection_list):
        ax.text(0.5, 5.5 - i*0.35, arrow, fontsize=10, color='red', fontweight='bold')

        ax.text(1.2, 5.5 - i*0.35, desc, fontsize=10, color=color, fontweight='bold' if color=='orange' else 'normal')
    

    legend_elements = [
        patches.Patch(color=langgraph_color, label='LangGraph Components'),
        patches.Patch(color=streamlit_color, label='Streamlit Decisions', alpha=0.7),
        patches.Patch(color=process_color, label='Processing Functions'),
        patches.Patch(color=rag_color, label='RAG Components'),
        patches.Patch(color=section_color, label='Workflow Sections', alpha=0.7),
        patches.Patch(color='black', label='Direct Flow Arrows'),
        patches.Patch(color='purple', label='RAG Internal Arrows'),
        patches.Patch(color='red', label='Numbered Connection Points')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), fontsize=8)
    
    ax.text(0.5, 3.2, 'Key Architecture Insights:', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.7))
    ax.text(0.5, 2.8, 'Black arrows = direct linear flow', fontsize=9)
    ax.text(0.5, 2.5, 'Purple arrows = RAG internal connections', fontsize=9)
    ax.text(0.5, 2.2, 'Red letters = numbered connection points for long paths', fontsize=9)
    ax.text(0.5, 1.9, 'Streamlit state management controls the workflow', fontsize=9)
    ax.text(0.5, 1.6, 'LangGraph handles only basic classifier->reasoning', fontsize=9)
    ax.text(0.5, 1.3, 'User decisions drive flow via button clicks', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('hybrid_compliance_workflow.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("Hybrid workflow diagram saved as 'hybrid_compliance_workflow.png'")
    plt.show()

if __name__ == "__main__":
    create_accurate_compliance_workflow() 
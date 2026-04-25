
dataset_order = [
    'taiwan',
    'plantvillage',
    'zpdd',
]

# reorder the datasets in the DataFrame
df['dataset'] = pd.Categorical(df['dataset'], categories=dataset_order, ordered=True)

# Calculate average student performance for each teacher group
dist_students = df[(df['type'] == 'Student') & (df['training_method'] == 'Distilled')]
avg_perf_by_teacher = dist_students.groupby(['dataset', 'teacher_id_for_distill'])['fp32_accuracy'].mean().reset_index()
avg_perf_by_teacher.rename(columns={'teacher_id_for_distill': 'method'}, inplace=True)

# Calculate average performance for the independent group
ind_students = df[(df['type'] == 'Student') & (df['training_method'] == 'Independent')]
avg_perf_independent = ind_students.groupby('dataset')['fp32_accuracy'].mean().reset_index()
avg_perf_independent['method'] = 'Independent'

# Combine and rank
teacher_perf_df = pd.concat([avg_perf_by_teacher, avg_perf_independent], ignore_index=True)
teacher_perf_df['method'] = teacher_perf_df['method'].str.replace('_GMN', '')
teacher_perf_df['rank'] = teacher_perf_df.groupby('dataset')['fp32_accuracy'].rank(method='min', ascending=False)

# Plotting
fig1, ax1 = plt.subplots(figsize=(12, 7))
sns.lineplot(
    data=teacher_perf_df,
    x='dataset',
    y='rank',
    hue='method',
    style='method',
    markers=True,
    markersize=20,
    linewidth=5,
    ax=ax1
)
ax1.invert_yaxis() # Rank 1 is at the top
# ax1.set_title('Teacher Strategy Consistency: Average Rank of Resulting Students', fontsize=18, weight='bold')
ax1.set_ylabel('Performance Rank (1=Best)', fontsize=26)
ax1.set_xlabel('Dataset', fontsize=26)
ax1.set_yticks([1, 2, 3, 4])
ax1.tick_params(axis='both', which='major', labelsize=26)
ax1.set_xticklabels([   
     'Taiwan',
    'Plantvillage',
    'ZPDD'
    ], ha='center') 

ax1.legend(title='Teaching Method', fontsize=16,loc='center right')
plt.tight_layout()

plt.savefig('teacher_consistency.pdf', dpi=300, bbox_inches='tight')
plt.savefig('teacher_consistency.eps', dpi=300, bbox_inches='tight')
plt.show()

# --- Plot 2: Student Architecture Consistency Analysis ---

# Calculate average performance for each student architecture across all its training methods
all_students = df[df['type'] == 'Student']
avg_perf_by_student = all_students.groupby(['dataset', 'student_model'])['fp32_accuracy'].mean().reset_index()
avg_perf_by_student['rank'] = avg_perf_by_student.groupby('dataset')['fp32_accuracy'].rank(method='min', ascending=False)

# Plotting
fig2, ax2 = plt.subplots(figsize=(12, 7))
sns.lineplot(
    data=avg_perf_by_student,
    x='dataset',
    y='rank',
    hue='student_model',
    style='student_model',
    markers=True,
    markersize=20,
    linewidth=5,
    ax=ax2
)
ax2.invert_yaxis() # Rank 1 is at the top
# ax2.set_title('Student Architecture Consistency: Average Performance Rank', fontsize=18, weight='bold')
ax2.set_ylabel('Performance Rank (1=Best)', fontsize=26)
ax2.set_xlabel('Dataset', fontsize=26)
ax2.set_yticks([1, 2, 3])
ax2.set_xticklabels([   
     'Taiwan',
    'Plantvillage',
    'ZPDD'
    ], ha='center')  # Ensure datasets are in the correct order
# font size adjustments
ax2.tick_params(axis='both', which='major', labelsize=26)
ax2.legend(title='Student Architecture', fontsize=20, loc='lower right', bbox_to_anchor=(.95, 0.15))
plt.tight_layout()

plt.savefig('student_architecture_consistency.pdf', dpi=300, bbox_inches='tight')
plt.savefig('student_architecture_consistency.eps', dpi=300, bbox_inches='tight')



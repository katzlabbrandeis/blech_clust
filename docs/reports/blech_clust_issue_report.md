# blech_clust Issue Report

## Executive Summary

- **Total Issues**: 768
- **Open Issues**: 293 (38.2%)
- **Closed Issues**: 475 (61.8%)

---

## Issue Categories (Clustered by Type)

The following categories were identified by analyzing issue titles and descriptions:

| Category | Count | Open | Closed | Description |
|----------|-------|------|--------|-------------|
| Data Processing | 415 | 147 | 268 | Issues related to data processing pipelines |
| File I/O | 363 | 132 | 231 | File handling, paths, loading, saving |
| Feature Request | 357 | 119 | 238 | New features and enhancements |
| Unit Analysis | 317 | 142 | 175 | Unit/neuron analysis, spike detection |
| Bug/Fix | 245 | 84 | 161 | Bug reports and fixes |
| Plotting/Visualization | 180 | 79 | 101 | Plots, visualizations, figures |
| Clustering | 119 | 44 | 75 | Clustering algorithms and thresholding |
| Documentation | 113 | 36 | 77 | Documentation, guides, examples |
| EMG | 86 | 25 | 61 | Electromyography processing |
| Installation | 78 | 24 | 54 | Installation, setup, platforms |
| Performance | 68 | 24 | 44 | Performance optimization |
| Classifier | 46 | 22 | 24 | Classification logic |
| Other | 31 | 10 | 21 | Uncategorized issues |

---

## Detailed Clusters

### 1. Data Processing (Issues related to pipeline and data handling)

Key sub-areas:
- **blech_process.py**: Multiple issues about processing logic
- **blech_make_arrays.py**: Array generation and trial info
- **dig-in handling**: Digital input processing

### 2. Plotting & Visualization

Key sub-areas:
- **Rolling threshold plots**: Several issues about adding global threshold overlays
- **CAR correlation plots**: Quality assurance plots
- **Unit waveform plots**: Visualization of sorted units

### 3. EMG (Electromyography)

Major cluster with 86 issues including:
- EMG filtering and processing
- BSA pipeline integration
- Classifier integration

### 4. Documentation

119 issues, many related to:
- Migration guides
- Installation documentation
- API examples

### 5. Installation & Platform Support

80 issues covering:
- Linux-only support (most issues closed)
- Docker containerization (still open)
- Windows alternatives (explored)

---

## Potential Duplicate Issues

The following issues have very similar titles (potential duplicates or related work):

### Exact Duplicates (100% similarity)

| Issue [#1](https://github.com/katzlabbrandeis/blech_clust/pull/1) | Issue [#2](https://github.com/katzlabbrandeis/blech_clust/issues/2) | Title |
|----------|----------|-------|
| [#709](https://github.com/katzlabbrandeis/blech_clust/pull/709) | [#708](https://github.com/katzlabbrandeis/blech_clust/issues/708) | Implement GLM-based firing rate estimation |
| [#512](https://github.com/katzlabbrandeis/blech_clust/pull/512) | [#495](https://github.com/katzlabbrandeis/blech_clust/pull/495) | Organize ephys_data.py methods into submodules |

### High Similarity (>90%)

| Issue [#1](https://github.com/katzlabbrandeis/blech_clust/pull/1) | Issue [#2](https://github.com/katzlabbrandeis/blech_clust/issues/2) | Similarity | Title |
|----------|----------|------------|-------|
| [#332](https://github.com/katzlabbrandeis/blech_clust/pull/332) | [#328](https://github.com/katzlabbrandeis/blech_clust/issues/328) | 98% | Firing rate estimation using orthonormal basis |
| [#559](https://github.com/katzlabbrandeis/blech_clust/pull/559) | [#318](https://github.com/katzlabbrandeis/blech_clust/issues/318) | 96% | PSTH similarity metric for QA |
| [#632](https://github.com/katzlabbrandeis/blech_clust/pull/632) | [#629](https://github.com/katzlabbrandeis/blech_clust/issues/629) | 95% | Memory issues in blech_channel_profile |
| [#553](https://github.com/katzlabbrandeis/blech_clust/pull/553) | [#551](https://github.com/katzlabbrandeis/blech_clust/issues/551) | 95% | Frequency filtering when loading data |
| [#501](https://github.com/katzlabbrandeis/blech_clust/pull/501) | [#497](https://github.com/katzlabbrandeis/blech_clust/issues/497) | 93% | Error during blech_run_process.py |
| [#630](https://github.com/katzlabbrandeis/blech_clust/pull/630) | [#627](https://github.com/katzlabbrandeis/blech_clust/issues/627) | 93% | Remove hardcoded downsampling |
| [#614](https://github.com/katzlabbrandeis/blech_clust/issues/614) | [#557](https://github.com/katzlabbrandeis/blech_clust/issues/557) | 92% | Error running blech_process.py |
| [#116](https://github.com/katzlabbrandeis/blech_clust/issues/116) | [#88](https://github.com/katzlabbrandeis/blech_clust/issues/88)/91 | 90-93% | Parallel step directory input |

---

## Related Issues (By References)

### Most Referenced Issues

These issues are referenced by multiple other issues, suggesting high importance or related work:

| Issue # | Times Referenced | Title |
|---------|------------------|-------|
| [#568](https://github.com/katzlabbrandeis/blech_clust/issues/568) | 9 | Interactive viewer for raw data |
| [#678](https://github.com/katzlabbrandeis/blech_clust/issues/678) | 4 | Test sequencing in GitHub workflow |
| [#282](https://github.com/katzlabbrandeis/blech_clust/pull/282) | 4 | Framework to create test dataset |
| [#562](https://github.com/katzlabbrandeis/blech_clust/issues/562) | 4 | blech_exp_info.py path requirements |

### Issues With Most References To Others

These issues reference many other issues (indicating work spanning multiple areas):

| Issue # | References | Title |
|---------|------------|-------|
| [#770](https://github.com/katzlabbrandeis/blech_clust/issues/770) | 26 | Issue running blech_init.py |
| [#575](https://github.com/katzlabbrandeis/blech_clust/pull/575) | 9 | Interactive raw data viewer |
| [#294](https://github.com/katzlabbrandeis/blech_clust/pull/294) | 6 | Small datasets for testing |

---

## Label Distribution

| Label | Count | Description |
|-------|-------|-------------|
| blech_bot | 135 | Automated bot actions |
| under_development | 66 | Work in progress |
| high priority | 14 | Critical issues |
| install | 9 | Installation-related |
| bug | 7 | Confirmed bugs |
| enhancement | 5 | Feature enhancements |
| lgtm | 4 | Ready to merge |
| waiting for merge | 4 | Awaiting merge |

---

## Priority Recommendations

### High Priority Open Issues (by labels and content)

1. **[#771](https://github.com/katzlabbrandeis/blech_clust/issues/771)**: Investigate discrepancy in firing rate plots
2. **[#770](https://github.com/katzlabbrandeis/blech_clust/issues/770)**: Issue running blech_init.py
3. **[#768](https://github.com/katzlabbrandeis/blech_clust/pull/768)**: Refactor classifier probability loading logic
4. **[#767](https://github.com/katzlabbrandeis/blech_clust/issues/767)**: Find way to emphasize amplitude when clustering
5. **[#766](https://github.com/katzlabbrandeis/blech_clust/issues/766)**: Add dependency override flag for pipeline

### Issues That Might Be Duplicates (Action Recommended)

1. **[#709](https://github.com/katzlabbrandeis/blech_clust/pull/709) & [#708](https://github.com/katzlabbrandeis/blech_clust/issues/708)**: Both about GLM-based firing rate estimation - should be merged
2. **[#512](https://github.com/katzlabbrandeis/blech_clust/pull/512) & [#494](https://github.com/katzlabbrandeis/blech_clust/issues/494)/[#495](https://github.com/katzlabbrandeis/blech_clust/pull/495)**: Multiple issues about organizing ephys_data.py - should be consolidated
3. **[#632](https://github.com/katzlabbrandeis/blech_clust/pull/632) & [#629](https://github.com/katzlabbrandeis/blech_clust/issues/629)**: Both about memory issues in channel profile - should be merged
4. **[#614](https://github.com/katzlabbrandeis/blech_clust/issues/614) & [#557](https://github.com/katzlabbrandeis/blech_clust/issues/557)**: Both about errors running blech_process.py - should be consolidated

---

## Summary of Findings

1. **768 total issues** - Good sized project with active maintenance
2. **38% open rate** - Active project with ongoing development
3. **Major clusters**: Data processing, plotting, EMG, documentation
4. **Multiple potential duplicates** found - recommend consolidation
5. **Documentation is well-covered** - Many docs issues are closed
6. **Installation issues largely resolved** - Linux-only is documented

---

*Report generated from GitHub Issues API*
*Repository: katzlabbrandeis/blech_clust*

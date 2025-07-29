import logging
import requests
import urllib.parse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Shared keyword definitions for consistency across detection methods
TIMESERIES_KEYWORDS = ['time series', 'timeseries', 'temporal', 'forecast', 'forecasting', 'sequential', 'time-based', 
                      'stock', 'weather', 'climate', 'sales', 'financial', 'daily', 'hourly', 'monthly', 'yearly', 'seasonal']

DATA_TYPE_KEYWORDS = {
    'Time-Series': TIMESERIES_KEYWORDS,
    'Image': ['image', 'picture', 'visual', 'computer vision', 'photo', 'pixel', 'photograph', 'visual recognition'],
    'Text': ['text', 'document', 'nlp', 'natural language', 'corpus', 'linguistic', 'text analysis', 'document analysis'],
    'Multivariate': ['multivariate', 'multiple variables', 'multi-dimensional', 'several features', 'mixed data', 'multimodal'],
    'Sequential': ['sequence', 'sequential', 'ordered data', 'step by step', 'sequential analysis', 'ordered sequence'],
    'Spatio-Temporal': ['spatial', 'geographic', 'location', 'geo', 'coordinates']
}

TASK_KEYWORDS = {
    'Classification': ['classification', 'classify', 'predict class', 'category', 'label', 'binary', 'multi-class'],
    'Regression': ['regression', 'continuous', 'numeric prediction', 'estimate', 'predict value', 'forecast', 'forecasting'],
    'Clustering': ['clustering', 'grouping', 'unsupervised', 'cluster analysis'],
    'Recommendation-Systems': ['recommendation', 'recommender', 'collaborative filtering'],
    'Causal-Discovery': ['causal', 'causality', 'cause', 'effect', 'causal inference']
}

SUBJECT_KEYWORDS = {
    'Business': ['business', 'finance', 'marketing', 'sales', 'customer', 'retail', 'bank', 'economic', 'profit', 'revenue', 'stock', 'trading'],
    'Life-Sciences': ['biology', 'medical', 'health', 'disease', 'genetic', 'clinical', 'patient', 'hospital', 'drug', 'cancer'],
    'Physical-Sciences': ['physics', 'chemistry', 'astronomy', 'energy', 'particle', 'chemical', 'molecular', 'weather', 'climate', 'temperature', 'sensor'],
    'CS-Engineering': ['computer', 'software', 'algorithm', 'network', 'system', 'engineering', 'technology', 'robot', 'ai', 'machine'],
    'Social-Sciences': ['social', 'psychology', 'sociology', 'demographic', 'census', 'population', 'survey', 'behavior', 'education', 'student'],
    'Game': ['game', 'chess', 'poker', 'tic-tac-toe', 'connect', 'puzzle', 'strategy'],
    'Law': ['legal', 'law', 'court', 'judge', 'crime', 'criminal', 'justice']
}

@dataclass
class Dataset:
    """Data class representing a dataset recommendation"""
    title: str
    description: str
    source: str  # "sample", "uci"
    domain: str  # UCI official: "Tabular", "Time-Series", "Sequential", "Multivariate", etc.
    url: str
    download_url: str
    size_mb: Optional[float] = None
    rows: Optional[int] = None
    columns: Optional[int] = None
    file_format: str = "csv"
    tags: List[str] = None
    difficulty: str = "beginner"  # "beginner", "intermediate", "advanced"
    relevance_score: float = 0.0
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class DatasetRecommendationTool:
    """Tool for finding and recommending datasets from online sources"""
    
    def __init__(self):
        """Initialize the dataset recommendation tool"""
        self.sources = {
            "uci": UCIMLRepoSource(),        # UCI ML Repository scraping
            # "kaggle": KaggleDatasetSource()  # Kaggle dataset search
        }
        logger.info("🔍 Dataset recommendation tool initialized")
    
    def recommend_datasets(self, user_query: str, domain: str = "Tabular", max_results: int = 5) -> Dict[str, Any]:
        """
        Main method to get dataset recommendations
        
        Args:
            user_query: User's original query/request
            domain: Detected UCI domain (Tabular, Time-Series, Sequential, Multivariate)
            max_results: Maximum number of datasets to recommend
            
        Returns:
            Dictionary with formatted results for agent
        """
        try:
            # Override domain based on query indicators
            detected_domain = self._detect_domain_from_query(user_query)
            if detected_domain:
                logger.info(f"🎯 Domain override: '{domain}' -> '{detected_domain}' based on query")
                print(f"🎯 DOMAIN OVERRIDE: {domain} -> {detected_domain}")
                domain = detected_domain
            
            logger.info(f"🔍 Searching for datasets: query='{user_query}', domain='{domain}'")
            print(f"🔍 DATASET SEARCH: Query='{user_query}', Domain='{domain}'")
            
            # Collect datasets from all sources
            all_datasets = []
            
            for source_name, source in self.sources.items():
                try:
                    logger.info(f"🔎 Searching {source_name} with semantic matching...")
                    datasets = source.search_datasets(
                        keywords=[user_query],  # Pass full query for semantic analysis
                        domain=domain,
                        max_results=max_results
                    )
                    all_datasets.extend(datasets)
                    logger.info(f"✅ Found {len(datasets)} datasets from {source_name}")
                except Exception as e:
                    logger.warning(f"⚠️ Error searching {source_name}: {e}")
                    continue
            
            # Take top results (sources already return ranked results)
            top_datasets = all_datasets[:max_results]
            
            logger.info(f"🎯 Returning {len(top_datasets)} dataset recommendations")
            print(f"🎯 DATASET RESULTS: {len(top_datasets)} recommendations found")
            
            # Format for agent consumption
            return self._format_for_agent(top_datasets, domain)
            
        except Exception as e:
            logger.error(f"❌ Dataset recommendation error: {e}")
            return {
                "success": False,
                "training_result": f"## ❌ Dataset Recommendation Error\n\n{str(e)}\n\nTry loading your own data with `pd.read_csv('your_data.csv')`"
            }
    
    def _format_for_agent(self, datasets: List[Dataset], domain: str) -> Dict[str, Any]:
        """Format dataset recommendations for agent consumption"""
        try:
            if not datasets:
                return {
                    "success": True,
                    "training_result": "## 📊 No Suitable Datasets Found\n\nNo datasets found matching your criteria. Try loading your own data with `pd.read_csv('your_data.csv')`"
                }
            
            # Create formatted recommendation text  
            displayed_count = min(len(datasets), 5)
            result_text = f"## 📊 Recommended Datasets\n\nBased on your query, here are {displayed_count} relevant datasets:\n\n"
            
            for i, dataset in enumerate(datasets[:5], 1):  # Show top 5
                loading_code = self.generate_loading_code(dataset, "df")
                
                result_text += f"""### {i}. {dataset.title}
**Source:** {dataset.source.title()} | **Domain:** {dataset.domain.title()}
**Size:** {dataset.rows} rows × {dataset.columns} columns ({dataset.size_mb}MB)

{dataset.description}

**Loading Code:**
```python
{loading_code}
```

---

"""
            
            result_text += "\n**Next Steps:** Choose a dataset above, run the loading code, then retry your ML training request."
            
            return {
                "success": True,
                "training_result": result_text
            }
            
        except Exception as e:
            logger.error(f"Formatting error: {e}")
            return {
                "success": False,
                "training_result": f"## ❌ Formatting Error\n\n{str(e)}"
            }
    
    def _detect_domain_from_query(self, user_query: str) -> str:
        """Detect UCI domain from user query keywords"""
        query_lower = user_query.lower()
        
        # Check each data type for keyword matches
        for domain, keywords in DATA_TYPE_KEYWORDS.items():
            if any(keyword in query_lower for keyword in keywords):
                return domain
        
        # Default to None - don't override if no strong indicators
        return None

    def generate_loading_code(self, dataset: Dataset, variable_name: str = "df") -> str:
        """Generate code to load a recommended dataset"""
        try:
            if dataset.source == "uci":
                return f"""
# Load {dataset.title} from UCI ML Repository
import pandas as pd

# Visit the UCI page to find the actual data file URL and replace below
# {variable_name} = pd.read_csv('paste_actual_download_url_here')
# print(f"Dataset loaded: {{len({variable_name})}} rows, {{len({variable_name}.columns)}} columns")
# {variable_name}.head()

print("UCI Dataset Page: {dataset.url}")
print("Visit the page above to find and download the dataset files")"""
            
            else:
                return f"""
# Load {dataset.title} dataset
import pandas as pd

{variable_name} = pd.read_csv('{dataset.download_url}')
print(f"Dataset loaded: {{len({variable_name})}} rows, {{len({variable_name}.columns)}} columns")
{variable_name}.head()"""
                
        except Exception as e:
            logger.warning(f"Code generation error: {e}")
            return f"# Error generating loading code for {dataset.title}"


class UCIMLRepoSource:
    """Interface to UCI ML Repository with web scraping"""
    
    def __init__(self):
        self.base_url = "https://archive.ics.uci.edu"
        self.datasets_url = f"{self.base_url}/datasets"
        
    def search_datasets(self, keywords: List[str], domain: str, max_results: int = 5) -> List[Dataset]:
        """Search UCI ML Repository comprehensively to find best matching datasets"""
        try:
            import requests
            from bs4 import BeautifulSoup
            
            logger.info("🔍 Starting comprehensive UCI database search...")
            
            user_query = ' '.join(keywords).lower()
            all_datasets = []
            seen_titles = set()
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Comprehensive search strategy: Try multiple filter combinations systematically
            search_strategies = self._generate_comprehensive_search_strategies(user_query, domain)
            
            target_dataset_count = max_results * 20  # Target 20x results for good selection
            
            for i, (strategy_name, base_params) in enumerate(search_strategies, 1):
                try:
                    logger.info(f"📋 Strategy {i}: {strategy_name}")
                    
                    # For each strategy, try multiple pages to get more datasets
                    strategy_datasets = 0
                    for page in range(10):  # Keep 10 pages per strategy for good data volume
                        skip = page * 100
                        
                        # Build URL with pagination
                        if isinstance(base_params, str):  # It's already a URL
                            if '?' in base_params:
                                url = f"{base_params}&skip={skip}&take=100"
                            else:
                                url = f"{base_params}?skip={skip}&take=100&sort=desc&orderBy=NumHits"
                        else:  # It's parameters
                            data_type, task, subject = base_params
                            url = self._build_filter_url(data_type, task, subject, skip=skip, take=100)
                        
                        logger.debug(f"    Page {page+1}: {url}")
                        
                        response = requests.get(url, headers=headers, timeout=15)
                        response.raise_for_status()
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Get all dataset links from this page
                        dataset_links = soup.find_all('a', href=lambda x: x and '/dataset/' in str(x))
                        logger.debug(f"    Found {len(dataset_links)} dataset links on page {page+1}")
                        
                        if not dataset_links:  # No more datasets on this page
                            break
                        
                        # Process all datasets found on this page
                        page_datasets = 0
                        for link in dataset_links:
                            dataset = self._parse_and_score_dataset(link, user_query, seen_titles, domain)
                            if dataset:
                                all_datasets.append(dataset)
                                seen_titles.add(dataset.title)
                                strategy_datasets += 1
                                page_datasets += 1
                        
                        logger.debug(f"    Processed {page_datasets} new datasets from page {page+1}")
                    
                    logger.info(f"    Total from strategy: {strategy_datasets} datasets")
                    
                    # Stop early if we have enough datasets from effective strategies  
                    if len(all_datasets) >= target_dataset_count:
                        logger.info(f"📊 Found {len(all_datasets)} datasets, stopping early to avoid inefficient strategies")
                        break
                        
                except Exception as e:
                    logger.debug(f"Strategy {i} ({strategy_name}) failed: {e}")
                    continue
            
            logger.info(f"📊 Total datasets collected: {len(all_datasets)}")
            
            # Now filter datasets using strict criteria matching
            if all_datasets:
                logger.info(f"📊 Filtering {len(all_datasets)} datasets using strict criteria...")
                
                # Extract keywords from user query for filtering
                keywords = user_query.split()
                
                # Apply strict filtering using _matches_criteria
                filtered_datasets = []
                for dataset in all_datasets:
                    if self._matches_criteria(dataset, keywords, domain):
                        filtered_datasets.append(dataset)
                
                logger.info(f"✅ {len(filtered_datasets)} datasets passed criteria filter")
                
                if filtered_datasets:
                    # Take the first max_results from filtered datasets
                    final_datasets = filtered_datasets[:max_results]
                    
                    logger.info(f"🏆 Returning top {len(final_datasets)} filtered datasets:")
                    for i, ds in enumerate(final_datasets, 1):
                        logger.info(f"  {i}. {ds.title}")
                    
                    return final_datasets
                else:
                    logger.warning("❌ No datasets passed the strict criteria filter")
                    return []
            else:
                logger.warning("❌ No datasets found in comprehensive search")
                return []
            
        except Exception as e:
            logger.error(f"❌ UCI comprehensive search error: {e}")
            print(f"❌ UCI COMPREHENSIVE SEARCH ERROR: {e}")
            return []
    
    def _generate_comprehensive_search_strategies(self, user_query: str, domain: str) -> List[tuple]:
        """Generate optimized search strategies (7 efficient strategies instead of 11+)"""
        strategies = []
        
        # Auto-detect primary filters
        primary_data_type = self._detect_uci_data_type(user_query, domain)
        primary_task = self._detect_uci_task(user_query)
        primary_subject = self._detect_uci_subject(user_query)
        
        # Strategy 1: Query-specific search (most targeted)
        if primary_data_type or primary_task or primary_subject:
            params = (primary_data_type, primary_task, primary_subject)
            name = f"Query-specific ({primary_data_type}, {primary_task}, {primary_subject})"
            strategies.append((name, params))
        
        # Strategy 2: Primary data type + detected task (targeted)
        if primary_data_type and primary_task:
            params = (primary_data_type, primary_task, "")
            strategies.append((f"Primary: {primary_data_type} + {primary_task}", params))
        
        # Strategy 3: Primary data type alone (broader coverage)
        if primary_data_type:
            params = (primary_data_type, "", "")
            strategies.append((f"Primary DataType: {primary_data_type}", params))
        
        # Strategy 4: Secondary data type (backup if primary fails)
        relevant_data_types = []
        if domain == "Time-Series":
            relevant_data_types = ["Sequential", "Multivariate"]  # Exclude primary Time-Series
        elif domain in ["Multivariate", "Image", "Text"]:
            relevant_data_types = ["Tabular", "Text"] if primary_data_type != "Tabular" else ["Multivariate"]
        else:  # Tabular or other
            relevant_data_types = ["Multivariate"] if primary_data_type != "Multivariate" else ["Tabular"]
        
        if relevant_data_types:
            secondary_data_type = relevant_data_types[0]
            params = (secondary_data_type, "", "")
            strategies.append((f"Secondary DataType: {secondary_data_type}", params))
        
        # Strategy 5: Primary task + best data type (task-focused)
        if primary_task:
            best_data_type = primary_data_type or "Multivariate"  # Use primary or default to multivariate
            params = (best_data_type, primary_task, "")
            strategies.append((f"Task-focused: {primary_task} + {best_data_type}", params))
        
        # Strategy 6: Subject area (only if specifically detected, not generic)
        if primary_subject and primary_subject != "":
            params = ("", "", primary_subject)
            strategies.append((f"Subject: {primary_subject}", params))
        
        # Strategy 7: Fallback with best generic combination (not unfiltered)
        fallback_data_type = "Multivariate"  # Most common and broad
        fallback_task = "Classification"     # Most common task
        params = (fallback_data_type, fallback_task, "")
        strategies.append((f"Fallback: {fallback_data_type} + {fallback_task}", params))
        
        logger.info(f"🎯 Generated {len(strategies)} optimized search strategies")
        return strategies
    
    def _parse_and_score_dataset(self, link, user_query: str, seen_titles: set, search_domain: str = None):
        """Parse dataset, avoiding duplicates (scoring done later after filtering)"""
        try:
            # Only process links that have text content
            title_text = link.get_text().strip()
            if not title_text or len(title_text) < 2:
                return None
                
            # Skip duplicates
            if title_text in seen_titles:
                return None
            
            dataset = self._parse_dataset_card(link, search_domain)
            if dataset:
                # Don't calculate relevance score here - will be done after filtering
                dataset.relevance_score = 0.0  # Initialize to 0
                return dataset
            return None
        except Exception as e:
            logger.debug(f"Error parsing dataset: {e}")
            return None
    
    def _detect_uci_data_type(self, user_query: str, domain: str) -> str:
        """Auto-detect data type filter from user query and domain"""
        query_lower = user_query.lower()
        
        # Check each data type for keyword matches
        for data_type, keywords in DATA_TYPE_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                return data_type
        
        # Domain-based defaults using UCI labels
        if domain == 'Time-Series':
            return 'Time-Series'
        elif domain in ['Multivariate', 'Image', 'Text']:
            return 'Multivariate'
        else:
            return 'Multivariate'  # Default to multivariate for tabular
    
    def _detect_uci_task(self, user_query: str) -> str:
        """Auto-detect task filter from user query"""
        query_lower = user_query.lower()
        
        # Check each task type for keyword matches
        for task, keywords in TASK_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                return task
        
        # For timeseries queries without specific task, don't apply task filter
        if any(kw in query_lower for kw in ['timeseries', 'time series', 'temporal']) and not any(kw in query_lower for kw in ['classification', 'regression', 'clustering']):
            return ''  # No task filter for generic timeseries
        
        # For generic queries, don't default to classification to allow more variety
        generic_terms = ['recommend', 'data', 'dataset']
        if any(term in query_lower for term in generic_terms) and len(query_lower.split()) <= 4:
            return ''  # No task filter for short generic queries
        
        # Default to classification only if query seems classification-related
        return 'Classification'
    
    def _detect_uci_subject(self, user_query: str) -> str:
        """Auto-detect subject area filter from user query"""
        query_lower = user_query.lower()
        
        # Check each subject area for keyword matches
        for subject, keywords in SUBJECT_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                return subject
        
        # For generic queries, return empty to avoid always getting same subject
        generic_terms = ['recommend', 'data', 'dataset', 'classification', 'regression', 'timeseries']
        if all(term in query_lower for term in generic_terms):
            return ''
        
        # Default to none (no subject filter) for better variety
        return ''
    
    def _build_filter_url(self, data_type: str, task: str, subject: str, skip: int = 0, take: int = 100) -> str:
        """Build UCI filter URL using correct API format"""
        import urllib.parse
        
        base_url = "https://archive.ics.uci.edu/datasets"
        params = {
            'skip': skip,
            'take': take,  # Get up to 100 datasets per request
            'sort': 'desc',
            'orderBy': 'NumHits',  # Sort by popularity/hits
            'search': ''
        }
        
        # Add data type filter if detected (note: plural 'Types')
        if data_type:
            params['Types'] = data_type
        
        # Add task filter if detected (note: plural 'Tasks') 
        if task:
            params['Tasks'] = task
        
        # Add subject area filter if detected (note: plural 'Subjects')
        if subject:
            params['Subjects'] = subject
        
        # Build query string
        query_string = urllib.parse.urlencode(params)
        filter_url = f"{base_url}?{query_string}"
        
        return filter_url
    
    def _parse_dataset_card(self, card, expected_domain: str = None):
        """Parse a dataset card from UCI website"""
        try:
            # Extract dataset name and URL
            name = ""
            dataset_url = ""
            description = ""
            
            # Method 1: Look for dataset link
            dataset_link = card.find('a', href=lambda x: x and '/dataset/' in str(x))
            if dataset_link:
                name = dataset_link.get_text().strip()
                href = dataset_link.get('href')
                dataset_url = f"{self.base_url}{href}" if not href.startswith('http') else href
            
            # Method 2: If card IS the link
            elif card.name == 'a' and '/dataset/' in str(card.get('href', '')):
                name = card.get_text().strip()
                href = card.get('href')
                dataset_url = f"{self.base_url}{href}" if not href.startswith('http') else href
            
            if not name or not dataset_url:
                return None
            
            # Extract description/abstract
            desc_element = (
                card.find('p') or 
                card.find('div', class_=lambda x: x and 'abstract' in str(x).lower()) or
                card.find('div', class_=lambda x: x and 'description' in str(x).lower())
            )
            if desc_element:
                description = desc_element.get_text().strip()
            
            # Extract metadata (instances, features, task type)
            instances = 100  # default
            attributes = 10  # default
            task_text = ""
            data_types_text = "Tabular"  # default, will be improved below
            
            # Look for metadata text
            all_text = card.get_text().lower()
            
            # Extract numbers for instances/features
            import re
            instance_match = re.search(r'(\d+)\s*instances?', all_text)
            if instance_match:
                instances = int(instance_match.group(1))
                
            feature_match = re.search(r'(\d+)\s*features?', all_text)
            if feature_match:
                attributes = int(feature_match.group(1))
            
            # Extract task type
            if 'classification' in all_text:
                task_text = "Classification"
            elif 'regression' in all_text:
                task_text = "Regression"
            elif 'clustering' in all_text:
                task_text = "Clustering"
            else:
                task_text = "Classification"  # default
            
            # Determine domain (use expected_domain if we're filtering for specific type)
            if expected_domain and expected_domain != "Tabular":
                domain = expected_domain
            else:
                domain = self._determine_domain(data_types_text, task_text, name)
            
            # Create dataset
            dataset = Dataset(
                title=name,
                description=description or f"UCI ML Repository dataset for {task_text.lower()}",
                source="uci",
                domain=domain,
                url=dataset_url,
                download_url=self._construct_download_url(dataset_url, name),
                size_mb=round(instances * attributes * 0.001, 2),  # Rough estimate
                rows=instances,
                columns=attributes,
                tags=self._extract_tags(task_text, data_types_text, name),
                difficulty="beginner" if instances < 1000 else "intermediate"
            )
            
            logger.debug(f"📊 Parsed UCI dataset: {name}")
            return dataset
            
        except Exception as e:
            logger.debug(f"Card parsing error: {e}")
            return None
    
    def _determine_domain(self, data_types: str, task: str, name: str):
        """Use UCI's official domain from data_types, with enhanced fallback detection"""
        # Use UCI's official data type if available
        if data_types and data_types != "Tabular":
            uci_type = data_types.strip()
            # Return UCI's official domain labels directly
            if uci_type in ['Time-Series', 'Sequential', 'Multivariate', 'Univariate', 'Text', 'Image', 'Other']:
                return uci_type
        
        # Enhanced fallback detection using title, task, and context
        text_lower = f"{data_types} {task} {name}".lower()
        
        # Strong time-series indicators
        if any(indicator in text_lower for indicator in ['time series', 'timeseries', 'temporal', 'forecast', 'stock', 'weather', 'climate', 'sales', 'financial', 'daily', 'hourly', 'monthly', 'yearly', 'seasonal']):
            return "Time-Series"
        
        # Image indicators
        elif any(indicator in text_lower for indicator in ['image', 'vision', 'photo', 'picture', 'pixel', 'visual']):
            return "Image"
        
        # Text indicators
        elif any(indicator in text_lower for indicator in ['text', 'nlp', 'language', 'speech', 'document', 'corpus']):
            return "Text"
        
        # Sequential indicators
        elif any(indicator in text_lower for indicator in ['sequential', 'sequence', 'ordered']):
            return "Sequential"
        
        # Multivariate indicators
        elif any(indicator in text_lower for indicator in ['multivariate', 'mixed', 'multiple variables']):
            return "Multivariate"
        
        # Default to UCI's tabular
        return "Tabular"
    
    
    def _extract_tags(self, task: str, data_types: str, name: str):
        """Extract relevant tags from UCI dataset info using comprehensive categories"""
        tags = []
        text_lower = f"{task} {data_types} {name}".lower()
        
        # Enhanced task-based tags (using UCI categories)
        task_indicators = {
            'classification': ['classification', 'classify', 'predict class', 'category', 'label', 'binary', 'multi-class'],
            'regression': ['regression', 'continuous', 'numeric prediction', 'estimate', 'forecast', 'predict value'],
            'clustering': ['clustering', 'grouping', 'unsupervised', 'cluster analysis'],
            'recommendation': ['recommendation', 'recommender', 'collaborative filtering', 'content-based'],
            'causal-discovery': ['causal', 'causality', 'cause', 'effect', 'causal inference'],
            'feature-selection': ['feature selection', 'variable selection', 'dimensionality']
        }
        
        for tag, keywords in task_indicators.items():
            if any(keyword in text_lower for keyword in keywords):
                tags.append(tag)
        
        # Enhanced subject area tags (using UCI categories)
        subject_indicators = {
            'life-sciences': ['biology', 'medical', 'health', 'disease', 'genetic', 'clinical', 'patient', 'hospital', 'drug', 'cancer', 'heart', 'brain'],
            'physical-sciences': ['physics', 'chemistry', 'astronomy', 'energy', 'particle', 'chemical', 'molecular', 'quantum', 'weather', 'climate'],
            'cs-engineering': ['computer', 'software', 'algorithm', 'network', 'system', 'engineering', 'technology', 'robot', 'ai', 'machine learning'],
            'social-sciences': ['social', 'psychology', 'sociology', 'demographic', 'census', 'population', 'survey', 'behavior', 'education', 'student'],
            'business': ['business', 'finance', 'marketing', 'sales', 'customer', 'retail', 'bank', 'economic', 'profit', 'revenue', 'market'],
            'game': ['game', 'chess', 'poker', 'tic-tac-toe', 'connect', 'puzzle', 'strategy'],
            'law': ['legal', 'law', 'court', 'judge', 'crime', 'criminal', 'justice']
        }
        
        for tag, keywords in subject_indicators.items():
            if any(keyword in text_lower for keyword in keywords):
                tags.append(tag)
        
        # Enhanced data type tags (using UCI categories)
        data_type_indicators = {
            'multivariate': ['multivariate', 'multiple variables', 'multi-dimensional', 'several features'],
            'univariate': ['univariate', 'single variable', 'one dimension', 'single feature'],
            'sequential': ['sequential', 'sequence', 'ordered', 'series'],
            'time-series': ['time series', 'temporal', 'time-based', 'chronological', 'timeseries'],
            'text': ['text', 'document', 'corpus', 'natural language', 'nlp', 'linguistic'],
            'images': ['image', 'picture', 'visual', 'pixel', 'photo', 'computer vision'],
            'spatio-temporal': ['spatial', 'geographic', 'location', 'geo', 'coordinates']
        }
        
        for tag, keywords in data_type_indicators.items():
            if any(keyword in text_lower for keyword in keywords):
                tags.append(tag)
            
        return tags[:8]  # Limit to 8 tags for better coverage
    
    def _construct_download_url(self, dataset_url: str, name: str):
        """Use the actual dataset URL as download link"""
        # Return the actual UCI dataset page URL - users can find download links there
        _ = name  # Suppress unused warning
        return dataset_url
    
    def _matches_criteria(self, dataset: Dataset, keywords: List[str], domain: str):
        """Check if dataset matches search criteria"""
        if not keywords:
            return True
            
        # Check domain compatibility using UCI labels (strict matching)
        if domain != "Tabular" and dataset.domain != domain:
            return False
            
        # Check keyword matches
        searchable_text = f"{dataset.title} {dataset.description} {' '.join(dataset.tags)}".lower()
        query_text = ' '.join(keywords).lower()
        
        # Enhanced matching using shared keyword definitions
        important_matches = False
        
        # Check for any important matches using shared constants
        for category_dict in [TASK_KEYWORDS, SUBJECT_KEYWORDS, DATA_TYPE_KEYWORDS]:
            for category, keywords_list in category_dict.items():
                if any(kw in query_text for kw in keywords_list):
                    if any(kw in searchable_text for kw in keywords_list) or category.lower() in searchable_text:
                        important_matches = True
                        break
            if important_matches:
                break
        
        # General keyword matches
        keyword_matches = 0
        for keyword in keywords:
            if len(keyword) > 3 and keyword.lower() in searchable_text:
                keyword_matches += 1
        
        # More lenient matching - accept datasets with any reasonable match
        if important_matches:
            return True
        elif keyword_matches >= 1:  # Reduced from 2 to 1 keyword match
            return True
        elif domain in ["Tabular", "Multivariate"] and keyword_matches > 0:
            return True
        else:
            # For generic queries, accept more datasets
            generic_terms = ['recommend', 'dataset', 'data']
            if any(term in ' '.join(keywords) for term in generic_terms):
                return True
            return False
    

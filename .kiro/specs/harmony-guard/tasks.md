# Implementation Plan

- [x] 1. Set up project structure and core interfaces



  - Create directory structure for api/, core/, lpe/, model/, configs/, lexicon-pack/, scripts/, infra/, and tests/
  - Define base interfaces and data models for ProcessedText, LPEResult, ClassifierResult, ContextResult, and AggregatedResult
  - Create configuration management system for loading ensemble weights, policy profiles, and preprocessing rules
  - _Requirements: 1.3, 1.4, 8.1, 8.2_

- [x] 2. Implement text preprocessing pipeline







  - [x] 2.1 Create language identification module






    - Implement fastText-style language detection with character n-gram fallback
    - Add per-segment language identification for code-mixed content
    - Include confidence thresholding and language distribution calculation
    - _Requirements: 1.3, 1.4, 3.3_

  - [x] 2.2 Build text normalization engine


    - Implement Unicode NFKC normalization and zero-width character stripping
    - Create homoglyph normalization and diacritic folding functions
    - Add punctuation normalization and repeated character compression
    - _Requirements: 4.1, 4.3, 4.4_

  - [x] 2.3 Implement transliteration system


    - Create Romanized to native script transliteration for major Indic languages
    - Build native to Romanized mirror pass for Hinglish coverage
    - Add transliteration confidence scoring and variant generation
    - _Requirements: 1.4, 4.5_

  - [x] 2.4 Create obfuscation handling module


    - Implement leet speak detection and normalization using configurable tables
    - Add phonetic obfuscation handling with language-specific rules
    - Create elongation detection and normalization (e.g., "fuuuu" → "fu")
    - _Requirements: 4.1, 4.2, 4.3_

  - [x] 2.5 Build tokenization system


    - Implement emoji-aware tokenization preserving emoji boundaries
    - Add script-aware tokenization handling mixed scripts
    - Create token-to-span mapping for result highlighting
    - _Requirements: 3.1, 4.4_

  - [x] 2.6 Add PII masking for privacy compliance


    - Implement email, phone, and ID pattern detection
    - Create configurable masking rules for different PII types
    - Add masked logging support for audit compliance
    - _Requirements: 7.1, 7.2_

- [x] 3. Implement Lexicon & Pattern Engine (LPE)


  - [x] 3.1 Create lexicon loading and management system


    - Build YAML-based lexicon loader for multilingual dictionaries
    - Implement weighted entry system with category and severity mappings
    - Add morphological variant handling and fuzzy matching
    - _Requirements: 2.2, 2.3, 1.3_

  - [x] 3.2 Build pattern matching engine


    - Implement regex-based pattern matching for elongated and obfuscated terms
    - Create efficient trie-based lookup for large lexicons
    - Add context-aware matching to reduce false positives
    - _Requirements: 4.1, 4.2, 2.2_

  - [x] 3.3 Implement emoji and symbol analysis


    - Create emoji sentiment mapping and abuse detection
    - Add kaomoji and Unicode symbol analysis
    - Implement combination pattern detection (emoji + text)
    - _Requirements: 4.4, 2.2_

  - [x] 3.4 Build span extraction and confidence scoring


    - Implement precise span extraction for matched patterns
    - Create rule-based confidence scoring system
    - Add rule trace generation for explainability
    - _Requirements: 3.1, 3.2, 3.5_

- [x] 4. Implement Transformer Classifier






  - [x] 4.1 Create model loading and inference wrapper


    - Build HuggingFace-compatible model loader for XLM-R or equivalent
    - Implement efficient batched inference with proper tokenization
    - Add model caching and warm-up functionality
    - _Requirements: 1.1, 1.2, 1.5_

  - [x] 4.2 Implement multi-head classification system


    - Create separate prediction heads for abuse categories, corporate decisions, and severity
    - Implement multi-label classification with proper threshold handling
    - Add attention weight extraction for span highlighting
    - _Requirements: 2.1, 2.2, 2.3, 3.1_

  - [x] 4.3 Build model calibration and confidence estimation


    - Implement temperature scaling for probability calibration
    - Create uncertainty estimation using model confidence
    - Add per-language confidence adjustment based on training data
    - _Requirements: 3.4, 6.3_

  - [x] 4.4 Add model performance monitoring






    - Implement prediction distribution tracking
    - Create drift detection using statistical tests
    - Add model performance metrics collection
    - _Requirements: 6.3, 8.1_

- [ ] 5. Implement Intent/Context Layer





  - [x] 5.1 Create contextual analysis engine


    - Implement negation detection and scope analysis
    - Build quotation and third-party reference identification
    - Add safe-use context recognition (HR reporting, educational content)
    - _Requirements: 5.3, 2.1_

  - [x] 5.2 Build context-aware confidence adjustment


    - Create context modifier calculation for different abuse categories
    - Implement confidence down-weighting for safe contexts
    - Add recommended action generation based on context analysis
    - _Requirements: 5.3, 3.4_

- [x] 6. Implement Ensemble Aggregator





  - [x] 6.1 Create probability calibration system


    - Implement temperature scaling for individual component outputs
    - Build ensemble weight learning and optimization
    - Add cross-validation for weight tuning
    - _Requirements: 1.2, 3.4_

  - [x] 6.2 Build result aggregation and decision logic


    - Implement weighted combination of LPE, Classifier, and Intent outputs
    - Create final decision logic with confidence thresholding
    - Add span consolidation and ranking algorithms
    - _Requirements: 2.1, 3.1, 3.4_

  - [x] 6.3 Implement explanation generation


    - Create decision trace generation combining all component inputs
    - Build human-readable explanation text generation
    - Add rule and model contribution attribution
    - _Requirements: 3.2, 3.5_

- [x] 7. Implement Corporate Policy Layer





  - [x] 7.1 Create policy configuration system


    - Build YAML-based policy profile loader
    - Implement configurable thresholds by category and severity
    - Add tenant-specific policy override support
    - _Requirements: 5.1, 5.2, 5.5_

  - [x] 7.2 Build policy rule engine


    - Implement threshold-based blocking rules
    - Create safe context allowlist processing
    - Add department-specific override handling
    - _Requirements: 5.2, 5.3, 5.4_

  - [x] 7.3 Add policy trace and audit logging


    - Implement detailed policy decision logging
    - Create audit trail for policy rule applications
    - Add policy change tracking and versioning
    - _Requirements: 3.5, 6.5_


- [x] 8. Implement FastAPI service layer




  - [x] 8.1 Create core API endpoints


    - Implement POST /v1/analyze endpoint with request validation
    - Build structured JSON response generation
    - Add error handling and HTTP status code mapping
    - _Requirements: 8.1, 8.2, 1.1_

  - [x] 8.2 Build feedback and monitoring endpoints


    - Implement POST /v1/feedback endpoint for continuous learning
    - Create GET /v1/health and GET /v1/metrics endpoints
    - Add GET /v1/config endpoint for non-secret configuration
    - _Requirements: 8.3, 8.4, 6.1, 9.3_

  - [x] 8.3 Add request processing pipeline integration


    - Integrate all ensemble components into request processing flow
    - Implement proper error handling and graceful degradation
    - Add request correlation ID generation and logging
    - _Requirements: 1.1, 9.2, 9.4_

  - [x] 8.4 Implement authentication and rate limiting


    - Add tenant ID validation and routing
    - Implement rate limiting per tenant with configurable limits
    - Create request authentication and authorization
    - _Requirements: 7.4, 8.5_

- [ ] 9. Add observability and monitoring
  - [ ] 9.1 Implement Prometheus metrics collection
    - Create metrics for latency, throughput, error rates, and per-language confidence
    - Add decision distribution tracking and component performance metrics
    - Implement custom metrics for business KPIs (precision, recall estimates)
    - _Requirements: 9.1, 1.1, 1.2_

  - [ ] 9.2 Build structured logging system
    - Implement JSON-structured logging with request correlation
    - Add PII masking in logs with configurable redaction rules
    - Create log aggregation and searchability features
    - _Requirements: 9.2, 7.2_

  - [ ] 9.3 Create health check and readiness probes
    - Implement liveness probe checking service availability
    - Build readiness probe validating model loading and component health
    - Add graceful shutdown handling for Kubernetes deployments
    - _Requirements: 9.3, 9.5_

- [ ] 10. Implement feedback and continuous learning system
  - [ ] 10.1 Create feedback data collection and storage
    - Build feedback request processing and validation
    - Implement secure feedback data storage with retention policies
    - Add feedback data aggregation and analysis tools
    - _Requirements: 6.1, 6.5_

  - [ ] 10.2 Build active learning pipeline
    - Implement low-confidence prediction identification
    - Create human review queue management
    - Add feedback integration into model retraining pipeline
    - _Requirements: 6.2, 6.4_

  - [ ] 10.3 Add automated model retraining
    - Implement drift detection using statistical tests
    - Create automated retraining triggers based on performance thresholds
    - Add A/B testing framework for model validation before deployment
    - _Requirements: 6.3, 6.4_

- [ ] 11. Create deployment configuration
  - [ ] 11.1 Build Docker containerization
    - Create multi-stage Dockerfile with optimized model loading
    - Implement proper health checks and signal handling
    - Add configuration via environment variables and mounted configs
    - _Requirements: 9.4, 9.5_

- [ ] 12. Create comprehensive test suite
  - [ ] 12.1 Build unit tests for core components
    - Write unit tests for preprocessing pipeline functions
    - Create tests for lexicon matching and pattern detection
    - Add tests for model inference and aggregation logic
    - _Requirements: 1.1, 1.2, 2.2, 4.1_

  - [ ] 12.2 Implement integration and end-to-end tests
    - Create integration tests for complete analysis pipeline
    - Build golden dataset tests for precision and recall validation
    - Add performance tests for latency and throughput requirements
    - _Requirements: 1.1, 1.2, 1.5_

  - [ ] 12.3 Add security and compliance tests
    - Implement tests for PII masking and log redaction
    - Create authentication and authorization tests
    - Add input validation and injection prevention tests
    - _Requirements: 7.1, 7.2, 7.4_
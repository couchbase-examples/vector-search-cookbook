#!/bin/bash

# Script to rename folders from fts/gsi to search_based/query_based
# This script processes directories with fts/gsi structure

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to log messages
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to update frontmatter for search_based (formerly fts)
update_search_based_frontmatter() {
    local frontmatter_file="$1"
    
    if [[ ! -f "$frontmatter_file" ]]; then
        log_warn "Frontmatter file not found: $frontmatter_file"
        return 1
    fi
    
    log_info "Updating search_based frontmatter: $frontmatter_file"
    
    # Create a temporary file
    local temp_file=$(mktemp)
    
    # Read and update the frontmatter
    while IFS= read -r line; do
        # Update path: with-fts -> with-search-vector-index
        if [[ "$line" =~ ^path: ]]; then
            echo "$line" | sed 's/with-fts/with-search-vector-index/g'
        # Update title: FTS service -> Search Vector Index
        elif [[ "$line" =~ ^title: ]]; then
            echo "$line" | sed 's/using FTS service/with Search Vector Index/g' | sed 's/FTS service/Search Vector Index/g'
        # Update short_title: FTS service -> Search Vector Index
        elif [[ "$line" =~ ^short_title: ]]; then
            echo "$line" | sed 's/using FTS service/with Search Vector Index/g' | sed 's/FTS service/Search Vector Index/g'
        # Update description: FTS service -> Search Vector Index
        elif [[ "$line" =~ FTS\ service ]]; then
            echo "$line" | sed 's/FTS service/Search Vector Index/g' | sed 's/using FTS\./with Search Vector Index./g'
        # Update tags: FTS -> Search Vector Index
        elif [[ "$line" =~ ^\ \ -\ FTS$ ]]; then
            echo "  - Search Vector Index"
        else
            echo "$line"
        fi
    done < "$frontmatter_file" > "$temp_file"
    
    # Replace the original file
    mv "$temp_file" "$frontmatter_file"
    log_info "✓ Updated search_based frontmatter"
}

# Function to update frontmatter for query_based (formerly gsi)
update_query_based_frontmatter() {
    local frontmatter_file="$1"
    
    if [[ ! -f "$frontmatter_file" ]]; then
        log_warn "Frontmatter file not found: $frontmatter_file"
        return 1
    fi
    
    log_info "Updating query_based frontmatter: $frontmatter_file"
    
    # Create a temporary file
    local temp_file=$(mktemp)
    
    # Extract the base path for alt_paths
    local base_path=""
    local in_frontmatter=false
    local after_frontmatter=false
    
    while IFS= read -r line; do
        # Track if we're in the frontmatter section
        if [[ "$line" == "---" ]]; then
            if [[ "$in_frontmatter" == false ]]; then
                in_frontmatter=true
                echo "$line"
            else
                # End of frontmatter - add alt_paths before closing
                if [[ -n "$base_path" ]]; then
                    # Generate alt_paths based on the original path
                    local alt_path1=$(echo "$base_path" | sed 's/with-global-secondary-index/with-hyperscale-vector-index/g')
                    local alt_path2=$(echo "$base_path" | sed 's/with-global-secondary-index/with-composite-vector-index/g')
                    echo "alt_paths: [\"$alt_path1\", \"$alt_path2\"]"
                fi
                echo "$line"
                after_frontmatter=true
            fi
            continue
        fi
        
        if [[ "$after_frontmatter" == true ]]; then
            echo "$line"
            continue
        fi
        
        # Update path: with-global-secondary-index -> with-hyperscale-or-composite-vector-index
        if [[ "$line" =~ ^path: ]]; then
            base_path=$(echo "$line" | sed 's/^path: "\(.*\)"/\1/')
            echo "$line" | sed 's/with-global-secondary-index/with-hyperscale-or-composite-vector-index/g'
        # Update title: Focus on framework/service, add "Hyperscale and Composite Vector Index"
        elif [[ "$line" =~ ^title: ]]; then
            # Extract framework names and restructure
            local new_title=$(echo "$line" | sed 's/using GSI index/with Couchbase Hyperscale and Composite Vector Index/g' | sed 's/GSI index/Couchbase Hyperscale and Composite Vector Index/g')
            echo "$new_title"
        # Update short_title
        elif [[ "$line" =~ ^short_title: ]]; then
            local new_short_title=$(echo "$line" | sed 's/using GSI index/with Couchbase Hyperscale and Composite Vector Index/g' | sed 's/GSI index/Couchbase Hyperscale and Composite Vector Index/g')
            echo "$new_short_title"
        # Update description: GSI -> Hyperscale and Composite Vector Index
        elif [[ "$line" =~ using\ GSI ]]; then
            echo "$line" | sed 's/using GSI\./with Couchbase Hyperscale and Composite Vector Index./g' | sed 's/using GSI/with Couchbase Hyperscale and Composite Vector Index/g'
        # Update tags: GSI -> Hyperscale Vector Index and Composite Vector Index
        elif [[ "$line" =~ ^\ \ -\ GSI$ ]]; then
            echo "  - Hyperscale Vector Index"
            echo "  - Composite Vector Index"
        else
            echo "$line"
        fi
    done < "$frontmatter_file" > "$temp_file"
    
    # Replace the original file
    mv "$temp_file" "$frontmatter_file"
    log_info "✓ Updated query_based frontmatter"
}

# Function to rename folder
rename_folder() {
    local old_path="$1"
    local new_name="$2"
    local parent_dir=$(dirname "$old_path")
    local new_path="$parent_dir/$new_name"
    
    if [[ -d "$new_path" ]]; then
        log_warn "Target folder already exists: $new_path"
        return 1
    fi
    
    log_info "Renaming: $old_path -> $new_path"
    mv "$old_path" "$new_path"
    echo "$new_path"
}

# Main processing function
process_directory() {
    local base_dir="$1"
    
    log_info "Starting folder renaming and frontmatter updates in: $base_dir"
    
    # Find all directories with fts or gsi subdirectories
    find "$base_dir" -type d \( -name "fts" -o -name "gsi" \) | while read -r dir; do
        local index_type=$(basename "$dir")
        
        log_info "Found $index_type directory: $dir"
        
        # Update frontmatter before renaming
        local frontmatter_path="$dir/frontmatter.md"
        if [[ -f "$frontmatter_path" ]]; then
            if [[ "$index_type" == "fts" ]]; then
                update_search_based_frontmatter "$frontmatter_path"
            elif [[ "$index_type" == "gsi" ]]; then
                update_query_based_frontmatter "$frontmatter_path"
            fi
        else
            log_warn "No frontmatter.md found in: $dir"
        fi
        
        # Rename the folder
        if [[ "$index_type" == "fts" ]]; then
            rename_folder "$dir" "search_based"
        elif [[ "$index_type" == "gsi" ]]; then
            rename_folder "$dir" "query_based"
        fi
    done
    
    log_info "Folder renaming and frontmatter updates complete!"
}

# Main execution
if [[ $# -eq 0 ]]; then
    # Use current directory if no argument provided
    WORKSPACE_DIR="."
else
    WORKSPACE_DIR="$1"
fi

if [[ ! -d "$WORKSPACE_DIR" ]]; then
    log_error "Directory not found: $WORKSPACE_DIR"
    exit 1
fi

log_info "Starting migration script"
log_info "Working directory: $WORKSPACE_DIR"
log_info "This script will:"
log_info "  1. Rename fts folders to search_based"
log_info "  2. Rename gsi folders to query_based"
log_info "  3. Update frontmatter files with new paths, titles, and tags"
log_info "  4. Add alt_paths to query_based frontmatter files"

# Process the directory
process_directory "$WORKSPACE_DIR"

log_info "All done! ✓"

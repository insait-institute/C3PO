// Program main is a dependency pre-caching utility for the Go sandbox runtime.
//
// This program blank-imports a curated set of commonly-used Go libraries so that
// running `go build` inside the sandbox compiles and caches their build artifacts
// ahead of time. By pre-warming the Go build cache, subsequent user-submitted code
// that depends on any of these packages will compile significantly faster because
// the compiled objects are already present in the module cache.
//
// Imported library families include:
//   - Testing:       testify
//   - Uber:          atomic, automaxprocs, goleak, zap
//   - ORM:           gorm
//   - Observability: opentelemetry, opentracing
//   - Configuration: viper, cobra, pflag
//   - Serialization: yaml.v2, yaml.v3
//   - Logging:       logrus
//   - Golang-x:      oauth2, text
//   - Miscellaneous:  go-enry (language detection), go-openapi/inflect
//
// The main function itself is trivial (prints "123") because the real purpose
// of this program is the side-effect of compiling its dependency graph.
package main

import (
	"fmt"

	_ "github.com/go-openapi/inflect"
	// testify
	_ "github.com/stretchr/testify/assert"
	// uber libs
	_ "go.uber.org/atomic"
	_ "go.uber.org/automaxprocs"
	_ "go.uber.org/goleak"
	_ "go.uber.org/zap"

	// orm
	_ "gorm.io/gorm"
	// opentelemetry
	_ "go.opentelemetry.io/otel"
	_ "go.opentelemetry.io/otel/exporters/stdout/stdoutmetric"
	_ "go.opentelemetry.io/otel/metric"
	_ "go.opentelemetry.io/otel/sdk"
	_ "go.opentelemetry.io/otel/sdk/metric"
	_ "go.opentelemetry.io/otel/trace"

	// opentracing
	_ "github.com/opentracing/opentracing-go"
	// yaml
	_ "gopkg.in/yaml.v2"
	_ "gopkg.in/yaml.v3"

	// golang x
	_ "golang.org/x/oauth2"
	_ "golang.org/x/text"

	// viper
	_ "github.com/spf13/cobra"
	_ "github.com/spf13/pflag"
	_ "github.com/spf13/viper"

	// logger
	_ "github.com/sirupsen/logrus"

	// other libs
	_ "github.com/go-enry/go-enry/v2"
)

func main() {
	fmt.Println("123")
}
